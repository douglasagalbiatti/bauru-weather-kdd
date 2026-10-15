import os
import unittest
from unittest.mock import patch
from src.keep_alive_streamlit import DEFAULT_STREAMLIT_URL, get_target_url, validate_and_format_url


class TestStreamlitUrlCapture(unittest.TestCase):
    def setUp(self):
        # Garante que STREAMLIT_APP_URL não interfira nos testes se já estiver no ambiente
        self.original_env = os.environ.get("STREAMLIT_APP_URL")
        if "STREAMLIT_APP_URL" in os.environ:
            del os.environ["STREAMLIT_APP_URL"]

    def tearDown(self):
        if self.original_env is not None:
            os.environ["STREAMLIT_APP_URL"] = self.original_env
        elif "STREAMLIT_APP_URL" in os.environ:
            del os.environ["STREAMLIT_APP_URL"]

    def test_default_fallback_url(self):
        """Verifica se a URL padrão oficial é utilizada na ausência de parâmetros."""
        captured = get_target_url()
        self.assertEqual(captured, DEFAULT_STREAMLIT_URL)

    def test_capture_from_environment_variable(self):
        """Verifica a captura correta a partir da variável de ambiente STREAMLIT_APP_URL."""
        test_url = "https://custom-env-app.streamlit.app"
        with patch.dict(os.environ, {"STREAMLIT_APP_URL": test_url}):
            captured = get_target_url()
            self.assertEqual(captured, test_url)

    def test_cli_argument_overrides_environment_variable_and_default(self):
        """Verifica se o argumento CLI tem precedência sobre a variável de ambiente e o default."""
        env_url = "https://env-app.streamlit.app"
        cli_url = "https://cli-override.streamlit.app"
        with patch.dict(os.environ, {"STREAMLIT_APP_URL": env_url}):
            captured = get_target_url(cli_url=cli_url)
            self.assertEqual(captured, cli_url)

    def test_auto_prefix_https_when_missing_scheme(self):
        """Verifica se URLs sem o prefixo http/https recebem https:// automaticamente."""
        raw_url = "bauru-weather.streamlit.app"
        formatted = validate_and_format_url(raw_url)
        self.assertEqual(formatted, "https://bauru-weather.streamlit.app")

    def test_strip_whitespace_and_quotes(self):
        """Verifica remoção de espaços em branco e aspas acidentais comuns em configs de CI."""
        raw_url = "  'https://bauru-weather.streamlit.app/'  "
        formatted = validate_and_format_url(raw_url)
        self.assertEqual(formatted, "https://bauru-weather.streamlit.app/")

    def test_raise_error_when_url_missing_and_no_default(self):
        """Verifica se ValueError é disparado quando a URL não foi informada e não há default."""
        with self.assertRaises(ValueError) as context:
            get_target_url(default_url=None)
        self.assertIn("URL do Streamlit não encontrada", str(context.exception))

    def test_raise_error_when_url_empty_or_whitespace(self):
        """Verifica se ValueError é disparado quando a URL fornecida contém apenas espaços."""
        with self.assertRaises(ValueError) as context:
            validate_and_format_url("   ")
        self.assertIn("espaços em branco", str(context.exception))


if __name__ == "__main__":
    unittest.main()
