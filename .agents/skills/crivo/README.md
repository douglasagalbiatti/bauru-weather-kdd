# Crivo — o portão de qualidade antes de subir

**Crivo** é uma *skill* para o Claude Code. Ele revisa **o que você mudou** no seu código
antes de você publicar — acha bug escondido, buraco de segurança e complexidade
desnecessária — e te dá, no fim, um veredito claro: **pode subir ou não.**

Feito para quem constrói software, site, app, ERP, automação ou script **com a ajuda da
IA** e chega naquela dúvida: *"a IA escreveu tudo isso... mas está bom? está seguro? posso
subir sem passar vergonha?"*.

> "Passar no crivo" = peneirar com rigor antes de deixar passar. É exatamente o que ele faz.

## O que ele faz

- **Revisa só o que mudou** (o *diff*) — não enche você de reclamação sobre código antigo.
- Olha por **três lentes**: 🐞 bug/lógica, 🔒 segurança e ✂️ simplificação.
- **Filtra o ruído** — mostra só o que importa e diz quantos sinais descartou.
- Aponta cada achado com **arquivo:linha**, explica **por que importa** em português de
  gente e sugere a **correção**.
- Fecha com um **veredito**: ✅ PODE SUBIR / 🟡 SUBA COM RESSALVAS / 🔴 NÃO SUBA AINDA.
- Se você aprovar, **aplica as correções** — mas nunca commita nem sobe nada sozinho.

## Como instalar

1. Baixe e descompacte o arquivo `Crivo.zip`.
2. Copie a pasta `Crivo` para dentro de `.claude/skills/` no seu projeto:
   ```
   seu-projeto/
   └── .claude/
       └── skills/
           └── Crivo/
               ├── SKILL.md
               ├── scripts/
               └── references/
   ```
   (Ou para `~/.claude/skills/Crivo/` se quiser que ele valha em todos os seus projetos.)
3. Abra o Claude Code nesse projeto. Pronto — o Crivo é reconhecido sozinho.

## Como usar (você só conversa — a IA cuida do resto)

Na **primeira vez**, o Crivo faz uma configuração rápida (pergunta o nome do projeto, a
stack, seu nível de rigor e o que mais te preocupa) e guarda tudo localmente em `.crivo/`.

Depois, é só falar naturalmente:

- "**Revisa o que eu mudei**" — a revisão completa do que ainda não subiu.
- "**Isso aqui está seguro?**" — foco em segurança.
- "**Dá pra simplificar essa parte?**" — foco em enxugar código.
- "**Revisa esse arquivo `caminho/arquivo.js`**" — raio-x de um arquivo.
- "**Posso subir?**" — ele revisa e te dá o veredito.
- "**Pode corrigir os pontos 1 e 3**" — ele aplica as correções que você aprovou.

Você nunca precisa digitar comando de terminal — a IA monta tudo por baixo dos panos e te
responde em português.

## Privacidade

- **Nada sai da sua máquina.** A configuração e o histórico ficam em `.crivo/`, que é local
  e fica de fora do Git.
- O motor `scripts/crivo.py` usa **só a biblioteca padrão do Python 3** — não instala nada,
  não acessa a internet e **não altera nenhum arquivo** (ele só lê e organiza).
- O Crivo **nunca commita nem publica** por conta própria. Revisar é com ele; apertar o
  botão de subir é sempre seu.

## Requisitos

- Claude Code.
- Python 3 (já vem no macOS e na maioria dos Linux).
- Git no projeto (pra ler o *diff*). Para revisar Pull Request do GitHub, o `gh` (opcional).

## Licença

MIT — use, adapte e compartilhe à vontade.
