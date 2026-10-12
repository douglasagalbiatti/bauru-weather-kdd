# As lentes do Crivo

Estas são as "lentes" que você (a IA) usa para olhar o *diff*. Cada lente é uma lista
do que procurar. **Você não precisa aplicar todas em todo arquivo** — use o bom senso:
um arquivo de HTML/CSS pede a lente de segurança de front; um script que fala com banco
pede injeção de SQL; um trecho de lógica pede a lente de bug. Adapte à linguagem e ao que
mudou.

Regra que vale para as três lentes: **só reporte o que foi introduzido ou mexido na
mudança**, explique em português de gente **por que importa na prática**, e traga a
**correção**. Se não tiver certeza, marque como "a verificar" e diga o que provaria — nunca
invente.

---

## 🐞 Lente 1 — Bug / lógica (o código faz o que promete?)

Leia o trecho e pergunte: *"em que situação isso faz a coisa errada?"*

- **Casos de borda não tratados** — o que acontece com entrada **vazia**, **nula/None**,
  **zero**, **lista vazia**, texto muito grande, número negativo, data inválida? O código
  quebra ou faz besteira?
- **Condição invertida ou frouxa** — `if` que testa o contrário do que devia; `>` onde era
  `>=`; `and` onde era `or`; comparação frouxa (`==` em vez de `===` em JS).
- **Erro de contagem (off-by-one)** — laço que vai um a mais ou um a menos; índice
  `lista[tamanho]` que estoura; fatia que corta errado.
- **Assíncrono mal amarrado** — `await`/`.then()` esquecido (o código segue antes do
  resultado chegar); Promise sem tratamento de erro; `async` sem `await`.
- **Variável/estado errado** — usada antes de existir; sombreada (dois nomes iguais);
  mutação de algo que devia ser imutável; valor que "vaza" entre iterações do laço.
- **Retorno/caminho faltando** — função que em algum caminho não retorna nada; `switch`/`if`
  sem o "e se não for nenhum desses?" (o `else`/`default`).
- **Recurso não fechado** — arquivo, conexão, timer que abre e não fecha (vazamento).
- **Confiar em dado externo sem checar** — resposta de API, parâmetro de URL, campo de
  formulário usado direto sem validar se veio no formato esperado.
- **Dinheiro/número sensível em ponto flutuante** — cálculo de valor em `float` que gera
  centavo errado (some com `0.1 + 0.2`); prefira inteiro em centavos ou tipo decimal.

Para cada suspeita, se der, descreva em uma frase **o cenário concreto** que quebra
("se o cliente mandar o carrinho vazio, a linha 42 divide por zero e a página cai").

---

## 🔒 Lente 2 — Segurança (dá pra alguém abusar disso?)

O medo nº1 de quem sobe código feito com IA. Procure:

- **Segredo no código** — senha, chave de API, token, string de conexão escritos direto no
  arquivo. **Tem que sair pra variável de ambiente.** Se já foi commitado, o segredo é
  considerado vazado: precisa ser **trocado/revogado**, não só apagado.
- **Injeção de SQL** — consulta ao banco montada juntando texto com dado do usuário
  (`"... WHERE id = " + id`). Um usuário mal-intencionado quebra a consulta. **Use consulta
  parametrizada** (placeholders `?` / `:nome`), nunca concatenação.
- **Injeção de comando** — dado de fora entrando num comando de terminal (`os.system`,
  `exec`, `shell=True`). **Passe argumentos em lista, sem shell**, e valide a entrada.
- **XSS (script injetado na tela)** — texto do usuário jogado como HTML (`innerHTML`,
  `dangerouslySetInnerHTML`, `v-html`, `document.write`). Se o texto tiver `<script>`, roda
  no navegador de quem abre. **Escape/sanitize** o conteúdo.
- **`eval`/execução dinâmica** — rodar texto como código. Quase sempre há um jeito melhor;
  se o texto vier de fora, é porta aberta.
- **Autorização não checada** — a rota/ação confere se **este** usuário pode fazer **isto**?
  Ou qualquer um logado (ou nem logado) consegue? Cuidado com "trocar o id na URL e ver o
  dado de outro cliente".
- **Validação/entrada** — tamanho, tipo e formato do que vem de fora são checados? Upload de
  arquivo valida extensão/tamanho/tipo real?
- **Cripto/hash fraco** — `md5`/`sha1` pra senha (use `bcrypt`/`argon2`); aleatório fraco
  (`Math.random`) pra token/senha (use gerador seguro).
- **Conexão insegura** — verificação de certificado desligada (`verify=False`,
  `rejectUnauthorized:false`); `http` onde devia ser `https`.
- **Exposição** — mensagem de erro que vaza detalhe interno (stack trace, caminho, versão);
  log gravando senha/cartão/CPF; endpoint de debug/admin aberto.

Ao reportar, diga o **risco na prática** ("com isso, alguém consegue ler o pedido de outro
cliente só trocando o número na URL") — não só o nome técnico.

---

## ✂️ Lente 3 — Simplificação / manutenção (dá pra ficar mais simples e claro?)

Código que a IA escreveu tende a ficar verboso e repetido. Enxugar reduz bug **e** gasto de
token (menos código pra IA reler depois). Procure:

- **Duplicação (copiar-colar)** — o mesmo bloco/lógica repetido em dois lugares. Se mudar num,
  esquece no outro → bug. **Extraia pra uma função/trecho único.**
- **Função que faz coisa demais** — bloco gigante que valida, calcula, salva e responde tudo
  junto. Quebre em passos com nome. Regra de bolso: se você não consegue nomear a função em
  uma frase, ela faz demais.
- **Nome que não diz nada** — `x`, `data2`, `tmp`, `handle3`. Um bom nome economiza um
  comentário e evita erro.
- **Número/texto mágico** — `86400`, `0.07`, `"ATIVO"` soltos no meio da lógica. Dê um nome
  (constante) que explique o que é.
- **Erro engolido em silêncio** — `except: pass`, `catch {}` vazio. O problema some sem
  ninguém ver; vira bug fantasma. Ao menos registre o erro (log) ou trate de verdade.
- **Sobra de depuração** — `console.log`, `print` de teste, `debugger`, código comentado que
  ficou pra trás. Limpe antes de subir.
- **Aninhamento profundo** — `if` dentro de `if` dentro de `if`. Inverta a condição e saia
  cedo ("return early") pra deixar plano e legível.
- **Reinventar o pronto** — escrever à mão o que a linguagem/uma função já faz (formatar
  data, somar lista, checar se está vazio). Use o que já existe.
- **Comentário que mente** — comentário que descreve algo que o código não faz mais (mudou o
  código, esqueceu o comentário). Comentário desatualizado engana; corrija ou apague.

Simplificação é **sugestão**, quase nunca bloqueia a subida — mas some com dívida antes que
ela vire bug.

---

## Como escolher a profundidade

- **Rigor "rápida"** (padrão pra ir rápido): foque na Lente 1 (bug) e Lente 2 (segurança) e
  só reporte o que é **importante ou crítico**. Deixe a simplificação de fora, ou só cite de
  passagem.
- **Rigor "profunda"** (pente-fino): aplique as três lentes, faça o **olhar cego** (2º passe),
  e inclua as sugestões de simplificação. Use quando for código sensível (pagamento, login,
  dado de cliente, migração de banco) ou quando a pessoa pedir "capricha".

Ajuste também pela **maior preocupação** que a pessoa marcou na config: se ela disse que o
medo é "vazar segurança", puxe o peso pra Lente 2; se é "código inchado", valorize a Lente 3.
