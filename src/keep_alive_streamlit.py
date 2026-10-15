"""
Script de manutenção de atividade (Keep-Alive) para Streamlit Community Cloud.
Utiliza Selenium WebDriver em modo Headless para simular acesso real de usuário
e impedir/reverter o modo sleep da aplicação.
"""

import argparse
import logging
import os
import sys
import time
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("streamlit_keep_alive")


def validate_and_format_url(url: str) -> str:
    """
    Valida e normaliza a URL do Streamlit.
    - Remove espaços em branco e aspas acidentais.
    - Adiciona 'https://' caso o usuário tenha informado apenas o domínio.
    - Verifica se a estrutura de URL é válida.
    """
    if not url:
        raise ValueError("A URL fornecida está vazia.")

    cleaned_url = url.strip().strip("'\"")

    if not cleaned_url:
        raise ValueError("A URL fornecida contém apenas espaços em branco.")

    # Se não tiver esquema (http/https), prefixa com https://
    if not cleaned_url.startswith(("http://", "https://")):
        cleaned_url = f"https://{cleaned_url}"

    parsed = urlparse(cleaned_url)
    if not parsed.netloc:
        raise ValueError(f"URL inválida ou malformatada: '{url}'")

    return cleaned_url


DEFAULT_STREAMLIT_URL = "https://inteligenciaclimaticabauru.streamlit.app"


def get_target_url(
    cli_url: str | None = None,
    env_var_name: str = "STREAMLIT_APP_URL",
    default_url: str | None = DEFAULT_STREAMLIT_URL,
) -> str:
    """
    Captura e valida a URL alvo do Streamlit:
    1. Prioriza o argumento via linha de comando (--url), se fornecido.
    2. Caso contrário, busca na variável de ambiente especificada (padrão STREAMLIT_APP_URL).
    3. Caso nenhum seja fornecido, utiliza o fallback padrão em default_url.
    """
    env_val = os.environ.get(env_var_name, "").strip()
    candidate_url = cli_url or env_val or default_url
    if not candidate_url:
        raise ValueError(
            f"URL do Streamlit não encontrada! Defina a variável de ambiente '{env_var_name}' "
            "ou passe via linha de comando com '--url <URL>'."
        )

    return validate_and_format_url(candidate_url)


def create_headless_driver() -> webdriver.Chrome:
    """
    Configura o navegador Chrome para execução em servidores headless (GitHub Actions).
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (X-UA-Compatible; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # Selenium 4 gerencia automaticamente o ChromeDriver
    return webdriver.Chrome(options=chrome_options)


def wake_up_streamlit_app(url: str, timeout: int = 60) -> bool:
    """
    Navega até a aplicação Streamlit, trata a tela de sono se necessário e confirma o carregamento.
    """
    logger.info("Iniciando sessão do navegador com Selenium...")
    driver = None
    try:
        driver = create_headless_driver()
        logger.info(f"Navegando para o destino: {url}")
        driver.get(url)

        wait = WebDriverWait(driver, timeout)

        # 1. Verifica se a aplicação está em estado de 'Sleep'
        logger.info("Verificando se a aplicação está em modo sleep...")
        time.sleep(5)  # Pausa para carregamento inicial do DOM

        # Busca botões típicos de acordar aplicação no Streamlit Community Cloud
        sleep_buttons = driver.find_elements(
            By.XPATH,
            "//button[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'get this app back up')]"
        )

        if sleep_buttons and sleep_buttons[0].is_displayed():
            logger.info("Aplicação adormecida detectada! Clicando no botão 'Yes, get this app back up!'...")
            sleep_buttons[0].click()
            logger.info("Comando de inicialização enviado. Aguardando o boot do Streamlit...")

        # 2. Aguarda a presença do container principal da aplicação Streamlit
        # Isso garante que a conexão via WebSocket foi estabelecida com sucesso
        logger.info("Aguardando carregamento completo da interface do Streamlit...")
        wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="stAppViewContainer"], #root'))
        )

        time.sleep(8)  # Tempo para conclusão dos handshakes WebSocket e telemetria

        logger.info(f"Sucesso! Aplicação carregada e ativa. Título da página: '{driver.title}'")
        return True

    except TimeoutException:
        logger.error(f"Tempo limite ({timeout}s) excedido ao aguardar o carregamento da aplicação.")
        if driver:
            logger.info(f"Título da página no momento do timeout: '{driver.title}'")
        return False
    except WebDriverException as e:
        logger.error(f"Erro no WebDriver do Selenium: {e}")
        return False
    finally:
        if driver:
            logger.info("Finalizando sessão do navegador.")
            driver.quit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script Keep-Alive para aplicações Streamlit Community Cloud via Selenium."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL da aplicação Streamlit (se não informada, lê de STREAMLIT_APP_URL).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Tempo limite em segundos para aguardar o carregamento (padrão: 60s).",
    )

    args = parser.parse_args()

    try:
        target_url = get_target_url(cli_url=args.url)
        logger.info(f"URL alvo capturada e validada: {target_url}")
    except ValueError as err:
        logger.error(f"Erro de configuração da URL: {err}")
        sys.exit(1)

    success = wake_up_streamlit_app(target_url, timeout=args.timeout)
    if not success:
        logger.error("Falha ao manter/acordar a aplicação Streamlit.")
        sys.exit(1)

    logger.info("Processo concluído com êxito!")


if __name__ == "__main__":
    main()
