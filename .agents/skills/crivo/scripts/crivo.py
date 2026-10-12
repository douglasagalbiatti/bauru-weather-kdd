#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crivo — motor determinístico da revisão de código.

Só biblioteca padrão do Python 3. Não instala nada, não acessa a internet,
não altera nenhum arquivo. Ele NÃO decide o que é bug: ele só faz a parte
mecânica (ler o diff, contar, detectar linguagem e passar uma pré-varredura
de sinais óbvios) e entrega tudo organizado pra IA julgar com as lentes.

Uso (a IA monta o comando; o dono só conversa):

  # revisar o que mudou e ainda não subiu (padrão)
  git diff | python3 scripts/crivo.py revisar

  # revisar a branch atual contra a principal
  git diff main...HEAD | python3 scripts/crivo.py revisar --base main

  # revisar um diff já salvo em arquivo
  python3 scripts/crivo.py revisar --diff mudanca.patch

  # pré-varredura rápida (raio-x) de UM arquivo, sem diff
  python3 scripts/crivo.py raiox --arquivo caminho/do/arquivo.py

  # pré-varredura rápida do projeto inteiro (arquivos de código)
  python3 scripts/crivo.py raiox --repo .

  # só os sinais de SEGURANÇA da entrada
  git diff | python3 scripts/crivo.py seguranca

  # saída em JSON (pra IA consumir) em qualquer modo
  git diff | python3 scripts/crivo.py revisar --json

Sempre lê o diff do stdin quando não recebe --diff/--arquivo/--repo.
"""

import sys
import os
import re
import json
import argparse

# --------------------------------------------------------------------------
# Detecção de linguagem por extensão
# --------------------------------------------------------------------------
LINGUAGENS = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".jsx": "JavaScript (React)", ".ts": "TypeScript", ".tsx": "TypeScript (React)",
    ".php": "PHP", ".rb": "Ruby", ".go": "Go", ".java": "Java", ".kt": "Kotlin",
    ".cs": "C#", ".c": "C", ".h": "C/C++", ".cpp": "C++", ".cc": "C++",
    ".rs": "Rust", ".swift": "Swift", ".sql": "SQL", ".sh": "Shell", ".bash": "Shell",
    ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS", ".vue": "Vue",
    ".yml": "YAML", ".yaml": "YAML", ".json": "JSON", ".env": "Env", ".dart": "Dart",
    ".ex": "Elixir", ".exs": "Elixir", ".scala": "Scala", ".r": "R", ".lua": "Lua",
}

# extensões que valem revisão de código (raio-x --repo ignora o resto)
EXT_CODIGO = set(LINGUAGENS.keys()) - {".json", ".yml", ".yaml", ".html", ".htm", ".css", ".scss"}

# pastas e arquivos que nunca entram no raio-x de repo
IGNORAR_PASTAS = {
    "node_modules", ".git", "dist", "build", "out", "vendor", ".venv", "venv",
    "__pycache__", ".next", ".nuxt", "coverage", "target", ".idea", ".vscode",
    "bower_components", ".cache", "tmp", ".crivo", "site-packages",
}
IGNORAR_SUFIXOS = (".min.js", ".min.css", ".map", ".lock", ".svg", ".png", ".jpg",
                   ".jpeg", ".gif", ".pdf", ".ico", ".woff", ".woff2", ".ttf")

# --------------------------------------------------------------------------
# Regras da pré-varredura. Cada regra é um sinal para a IA OLHAR — não é um
# veredito. categoria, nome, regex, dica (por que olhar), linguagens (None = todas)
# --------------------------------------------------------------------------
Rule = lambda categoria, nome, padrao, dica, langs=None: {
    "categoria": categoria, "nome": nome,
    "regex": re.compile(padrao, re.IGNORECASE), "dica": dica, "langs": langs,
}

REGRAS = [
    # ---- SEGREDO: credencial que pode ter vazado pro código ----
    Rule("segredo", "Chave/token no código",
         r"""(api[_-]?key|secret|passwd|password|senha|token|access[_-]?key)\s*[:=]\s*['"][^'"]{6,}['"]""",
         "Parece uma senha/chave escrita direto no código. Segredo tem que vir de variável de ambiente, nunca ficar no arquivo."),
    Rule("segredo", "Chave AWS",
         r"""\bAKIA[0-9A-Z]{16}\b""",
         "Padrão de chave de acesso AWS. Se for real, revogue já e tire do código."),
    Rule("segredo", "Chave secreta de serviço",
         r"""\b(sk_live_|sk_test_|rk_live_|xox[baprs]-|ghp_|gho_|AIza[0-9A-Za-z\-_]{20,})""",
         "Formato de chave secreta de um serviço pago (Stripe, Slack, GitHub, Google). Não pode ir pro repositório."),
    Rule("segredo", "Chave privada",
         r"""-----BEGIN (RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----""",
         "Uma chave privada inteira dentro do código. Isso nunca deve ser versionado."),

    # ---- PERIGO: coisa que pode virar buraco de segurança / quebrar ----
    Rule("perigo", "Execução dinâmica (eval/exec)",
         r"""(?<![\w.])(eval|exec)\s*\(""",
         "Roda texto como se fosse código. Se esse texto vier do usuário, vira porta de entrada pra ataque. Quase sempre dá pra evitar.",
         {"Python", "JavaScript", "TypeScript", "JavaScript (React)", "TypeScript (React)", "PHP", "Ruby"}),
    Rule("perigo", "Comando no sistema operacional",
         r"""(os\.system|subprocess\.[a-z_]+\([^)]*shell\s*=\s*True|child_process\.(exec|execSync)|Runtime\.getRuntime\(\)\.exec|`[^`]*\$\{)""",
         "Monta um comando de terminal. Se algum pedaço vier de fora, dá pra injetar comando. Prefira passar argumentos em lista e sem shell."),
    Rule("perigo", "SQL montado por concatenação",
         r"""(SELECT|INSERT|UPDATE|DELETE)\b[^;]*?(\+\s*['"]?\s*(req|request|params|input|body|user|var|\$)|['"]\s*\+|%\s*\(|f['"].*\{)""",
         "Consulta ao banco montada juntando texto. É o caminho clássico da injeção de SQL. Use consulta parametrizada (com ? / placeholders)."),
    Rule("perigo", "HTML injetado sem escapar",
         r"""(dangerouslySetInnerHTML|\.innerHTML\s*=|document\.write\s*\(|v-html)""",
         "Joga texto direto como HTML. Se o texto tiver script, executa no navegador do usuário (XSS). Escape o conteúdo."),
    Rule("perigo", "Desserialização insegura",
         r"""(pickle\.loads|yaml\.load\s*\((?![^)]*Safe)|marshal\.loads|unserialize\s*\(|ObjectInputStream)""",
         "Transforma dado de fora em objeto sem trava. Dá pra usar isso pra rodar código. Use a versão 'safe' (ex: yaml.safe_load)."),
    Rule("perigo", "Verificação de certificado desligada",
         r"""(verify\s*=\s*False|rejectUnauthorized\s*:\s*false|InsecureSkipVerify\s*:\s*true|CURLOPT_SSL_VERIFYPEER\s*,\s*(0|false))""",
         "Desliga a checagem do certificado da conexão segura. Abre espaço pra alguém interceptar os dados. Só em teste, nunca em produção."),
    Rule("perigo", "Hash fraco pra senha",
         r"""\b(md5|sha1)\s*\(""",
         "MD5/SHA1 são fracos pra guardar senha. Use bcrypt/argon2/scrypt se isso lida com senha."),

    # ---- DEBUG: sobra de desenvolvimento que não devia subir ----
    Rule("debug", "Sobra de depuração",
         r"""(?<![\w.])(console\.(log|debug|dir)|debugger|var_dump|dd\s*\(|print_r|System\.out\.println|fmt\.Println)\s*[\(;]""",
         "Parece rastro de teste (log/print de depuração). Costuma sujar a saída em produção e às vezes vaza dado. Confirme se deve sair."),
    Rule("debug", "Marcador de pendência",
         r"""(?<![\w])(TODO|FIXME|XXX|HACK|GAMBIARRA|WIP)\b""",
         "Marcador de 'terminar depois'. Vale checar se isso é uma pendência que não pode subir agora."),

    # ---- FRÁGIL: cheiro de código que quebra fácil / difícil de manter ----
    Rule("fragil", "Erro engolido em silêncio",
         r"""(except\s*:\s*(pass|\.\.\.)|except\s+Exception\s*:\s*pass|catch\s*\([^)]*\)\s*\{\s*\}|catch\s*\{\s*\})""",
         "Captura o erro e não faz nada. O problema some sem ninguém ver — depois vira bug fantasma. Ao menos registre o erro."),
    Rule("fragil", "Captura de erro ampla demais",
         r"""^\s*except\s*:\s*(#.*)?$""",
         "Um 'except:' pelado captura QUALQUER erro (até Ctrl+C do sistema) e costuma esconder problema de verdade. Diga qual erro você trata e registre-o.",
         {"Python"}),
    Rule("fragil", "Comparação frouxa (== / !=)",
         r"""[^=!<>]=[=]([^=]|$)|[^=!<>]!=([^=]|$)""",
         "Em JS/PHP, == compara 'mais ou menos' e causa surpresa (0 == '' vira verdadeiro). Prefira === / !==.",
         {"JavaScript", "TypeScript", "JavaScript (React)", "TypeScript (React)", "PHP"}),
    Rule("fragil", "Tipo 'any' (TypeScript)",
         r""":\s*any\b|<any>|as\s+any\b""",
         "O 'any' desliga a checagem de tipo naquele ponto — o TypeScript para de te proteger ali. Vale tipar de verdade.",
         {"TypeScript", "TypeScript (React)"}),
    Rule("fragil", "Número mágico em condição",
         r"""(if|while|elif|else if)\s*\([^)]*[<>=!]=?\s*\d{3,}""",
         "Número solto no meio da lógica sem nome. Daqui a um mês ninguém lembra o que 86400 significa. Vale dar um nome (constante)."),
]


# --------------------------------------------------------------------------
# Parse de diff unificado -> por arquivo, linhas ADICIONADAS (com nº novo)
# --------------------------------------------------------------------------
def parse_diff(texto):
    arquivos = {}          # caminho -> lista de (num_linha_novo, conteudo)
    caminho = None
    novo_ln = 0
    dentro_hunk = False
    for linha in texto.splitlines():
        if linha.startswith("diff --git"):
            caminho = None
            dentro_hunk = False
            continue
        if linha.startswith("+++ "):
            p = linha[4:].strip()
            if p.startswith("b/"):
                p = p[2:]
            caminho = None if p == "/dev/null" else p
            arquivos.setdefault(caminho, [])
            continue
        m = re.match(r"@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", linha)
        if m:
            novo_ln = int(m.group(1))
            dentro_hunk = True
            continue
        if not dentro_hunk or caminho is None:
            continue
        if linha.startswith("+") and not linha.startswith("+++"):
            arquivos.setdefault(caminho, []).append((novo_ln, linha[1:]))
            novo_ln += 1
        elif linha.startswith("-") and not linha.startswith("---"):
            pass  # linha removida não conta linha nova
        else:
            novo_ln += 1
    # remove entradas None/vazias
    return {k: v for k, v in arquivos.items() if k}


def lang_de(caminho):
    _, ext = os.path.splitext(caminho.lower())
    return LINGUAGENS.get(ext, "Outro")


def limpo_para_regra(linha):
    """Tira o conteúdo dentro de comentário de fim-de-linha simples, pra
    reduzir falso-positivo de comparação/números dentro de comentário."""
    return linha


def aplica_regras(caminho, linhas):
    """linhas: lista de (num, conteudo). Retorna lista de achados."""
    lang = lang_de(caminho)
    achados = []
    for num, conteudo in linhas:
        texto = conteudo.rstrip("\n")
        stripado = texto.strip()
        # pula linha em branco
        if not stripado:
            continue
        # heurística: pula linha que é claramente comentário puro (pra regra de número/comparação)
        eh_comentario = bool(re.match(r"""^\s*(#|//|\*|/\*|<!--|--)""", texto))
        for regra in REGRAS:
            if regra["langs"] is not None and lang not in regra["langs"]:
                continue
            # regras de 'fragil' de estilo não valem dentro de comentário
            if eh_comentario and regra["categoria"] in ("fragil",):
                continue
            if regra["regex"].search(texto):
                achados.append({
                    "categoria": regra["categoria"],
                    "regra": regra["nome"],
                    "arquivo": caminho,
                    "linha": num,
                    "trecho": stripado[:200],
                    "dica": regra["dica"],
                    "linguagem": lang,
                })
    return achados


# --------------------------------------------------------------------------
# Coleta de entrada
# --------------------------------------------------------------------------
def ler_stdin():
    if sys.stdin is None or sys.stdin.isatty():
        return ""
    try:
        return sys.stdin.read()
    except Exception:
        return ""


def linhas_de_arquivo(caminho):
    try:
        with open(caminho, "r", encoding="utf-8", errors="replace") as f:
            return [(i + 1, l.rstrip("\n")) for i, l in enumerate(f)]
    except Exception as e:
        return None


def varre_repo(raiz):
    """Retorna {caminho: [(num, conteudo)]} pros arquivos de código do repo."""
    resultado = {}
    for base, dirs, files in os.walk(raiz):
        dirs[:] = [d for d in dirs if d not in IGNORAR_PASTAS and not d.startswith(".")]
        for nome in files:
            if nome.endswith(IGNORAR_SUFIXOS):
                continue
            _, ext = os.path.splitext(nome.lower())
            if ext not in EXT_CODIGO:
                continue
            caminho = os.path.join(base, nome)
            rel = os.path.relpath(caminho, raiz)
            try:
                if os.path.getsize(caminho) > 400_000:
                    continue
            except OSError:
                continue
            linhas = linhas_de_arquivo(caminho)
            if linhas is not None:
                resultado[rel] = linhas
    return resultado


# --------------------------------------------------------------------------
# Saída
# --------------------------------------------------------------------------
EMOJI = {"segredo": "🔑", "perigo": "⚠️", "debug": "🧹", "fragil": "🩹"}
NOME_CAT = {
    "segredo": "SEGREDO exposto",
    "perigo": "PERIGO (segurança/quebra)",
    "debug": "SOBRA de depuração",
    "fragil": "FRÁGIL (cheiro de código)",
}
ORDEM_CAT = ["segredo", "perigo", "fragil", "debug"]


def monta_resumo(arquivos, achados, modo):
    langs = {}
    total_linhas = 0
    for caminho, linhas in arquivos.items():
        total_linhas += len(linhas)
        l = lang_de(caminho)
        langs[l] = langs.get(l, 0) + 1
    por_cat = {}
    for a in achados:
        por_cat[a["categoria"]] = por_cat.get(a["categoria"], 0) + 1
    return {
        "modo": modo,
        "arquivos": len(arquivos),
        "linhas_analisadas": total_linhas,
        "linguagens": langs,
        "sinais_encontrados": len(achados),
        "por_categoria": por_cat,
    }


def imprime_markdown(resumo, achados, modo):
    print("# Crivo — pré-varredura (parte mecânica)\n")
    print("> Estes são **sinais para a IA olhar**, não vereditos. A IA ainda")
    print("> vai julgar cada um com as lentes, cortar falso-positivo e escrever")
    print("> o parecer final. Nada aqui foi confirmado como erro ainda.\n")
    print(f"- **Modo:** {modo}")
    print(f"- **Arquivos:** {resumo['arquivos']}  |  **Linhas analisadas:** {resumo['linhas_analisadas']}")
    if resumo["linguagens"]:
        langs = ", ".join(f"{k} ({v})" for k, v in sorted(resumo["linguagens"].items(), key=lambda x: -x[1]))
        print(f"- **Linguagens:** {langs}")
    print(f"- **Sinais na pré-varredura:** {resumo['sinais_encontrados']}")
    if resumo["por_categoria"]:
        cats = ", ".join(f"{EMOJI.get(k,'')} {k}: {v}" for k, v in resumo["por_categoria"].items())
        print(f"- **Por categoria:** {cats}")
    print()

    if not achados:
        print("Nenhum sinal óbvio na pré-varredura mecânica. **Isso não quer dizer")
        print("que está tudo certo** — a pré-varredura só pega padrões conhecidos.")
        print("A revisão de verdade (lógica, bug, arquitetura) é a IA que faz, lendo o diff.")
        return

    # agrupa por categoria em ordem de gravidade
    for cat in ORDEM_CAT:
        do_cat = [a for a in achados if a["categoria"] == cat]
        if not do_cat:
            continue
        print(f"## {EMOJI.get(cat,'')} {NOME_CAT.get(cat, cat)} — {len(do_cat)} sinal(is)\n")
        for a in do_cat:
            print(f"- **{a['arquivo']}:{a['linha']}** — {a['regra']}")
            print(f"  - trecho: `{a['trecho']}`")
            print(f"  - por que olhar: {a['dica']}")
        print()


def main():
    ap = argparse.ArgumentParser(add_help=True, description="Crivo — motor da revisão de código")
    ap.add_argument("modo", nargs="?", default="revisar",
                    choices=["revisar", "seguranca", "simplificar", "raiox"],
                    help="revisar (tudo) | seguranca (só segurança) | simplificar (só cheiro) | raiox (varredura rápida)")
    ap.add_argument("--diff", help="caminho de um arquivo .patch/.diff (senão lê do stdin)")
    ap.add_argument("--arquivo", help="raio-x de UM arquivo (sem diff)")
    ap.add_argument("--repo", help="raio-x do projeto inteiro a partir desta pasta")
    ap.add_argument("--base", help="rótulo da branch base (só informativo no relatório)")
    ap.add_argument("--json", action="store_true", help="saída em JSON pra IA consumir")
    args = ap.parse_args()

    # 1) monta o conjunto {caminho: [(num, conteudo)]}
    if args.repo:
        if not os.path.isdir(args.repo):
            print(f"ERRO: pasta não encontrada: {args.repo}", file=sys.stderr)
            sys.exit(2)
        arquivos = varre_repo(args.repo)
        origem = f"raio-x do repositório ({args.repo})"
    elif args.arquivo:
        if not os.path.isfile(args.arquivo):
            print(f"ERRO: arquivo não encontrado: {args.arquivo}", file=sys.stderr)
            sys.exit(2)
        linhas = linhas_de_arquivo(args.arquivo)
        arquivos = {args.arquivo: linhas or []}
        origem = f"raio-x do arquivo ({args.arquivo})"
    else:
        if args.diff:
            try:
                texto = open(args.diff, "r", encoding="utf-8", errors="replace").read()
            except Exception as e:
                print(f"ERRO ao ler o diff: {e}", file=sys.stderr)
                sys.exit(2)
        else:
            texto = ler_stdin()
        if not texto.strip():
            print("ERRO: não recebi nenhum diff. Rode `git diff | python3 scripts/crivo.py revisar`,",
                  file=sys.stderr)
            print("      ou passe --diff arquivo.patch, --arquivo caminho, ou --repo .", file=sys.stderr)
            sys.exit(1)
        arquivos = parse_diff(texto)
        base = f" (base: {args.base})" if args.base else ""
        origem = f"diff das mudanças{base}"

    if not arquivos:
        print("ERRO: não encontrei nenhum arquivo com mudança pra analisar.", file=sys.stderr)
        sys.exit(1)

    # 2) roda as regras
    todos = []
    for caminho, linhas in arquivos.items():
        todos.extend(aplica_regras(caminho, linhas))

    # 3) filtra por modo
    if args.modo == "seguranca":
        todos = [a for a in todos if a["categoria"] in ("segredo", "perigo")]
    elif args.modo == "simplificar":
        todos = [a for a in todos if a["categoria"] in ("fragil", "debug")]
    # revisar e raiox: mantém tudo

    resumo = monta_resumo(arquivos, todos, args.modo)
    resumo["origem"] = origem

    # 4) imprime
    if args.json:
        print(json.dumps({"resumo": resumo, "achados": todos, "arquivos_alterados": list(arquivos.keys())},
                         ensure_ascii=False, indent=2))
    else:
        imprime_markdown(resumo, todos, origem)


if __name__ == "__main__":
    main()
