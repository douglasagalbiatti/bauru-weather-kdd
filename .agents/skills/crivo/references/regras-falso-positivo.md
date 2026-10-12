# Regras de falso-positivo — o que NÃO reportar

Um revisor que grita por tudo perde a confiança e é ignorado. O valor do Crivo está em
**filtrar o ruído** e mostrar só o que importa. Antes de reportar qualquer achado, passe
ele por estes filtros. Se cair em qualquer um, **não reporte** (ou rebaixe pra "boa prática",
sem alarde).

## Sempre filtre

1. **Problema pré-existente (não é da mudança)**
   - O trecho já existia antes e a pessoa não mexeu nele. A régua do Crivo é o *diff*.
   - Na dúvida se a linha é nova, dá pra conferir com `git blame -L <linha>,<linha> <arquivo>`.
   - Se é importante mas antigo, no máximo mencione **de leve** no fim ("fora do escopo desta
     mudança, mas notei X no código existente") — nunca como bloqueio da subida.

2. **Território do editor/linter/compilador**
   - Erro de sintaxe, tipo trocado (o TypeScript pega), variável não usada, ordem de import,
     ponto-e-vírgula, indentação. A ferramenta da pessoa já cuida disso. Não gaste bala aí.

3. **Nitpick sem impacto real**
   - Pergunte-se: *"um engenheiro sênior perderia tempo comentando isso num code review?"*
   - "Poderia ser um pouco melhor" **não** é motivo pra reportar. Só reporte o que é
     **acionável e com consequência**.

4. **Já silenciado de propósito**
   - Linhas com `// eslint-disable`, `# noqa`, `// nolint`, `#[allow(...)]`,
     `@SuppressWarnings`, `// @ts-ignore` — a pessoa marcou que sabe. Respeite (a não ser que
     esteja silenciando algo perigoso de verdade — aí vale um alerta gentil).

5. **Mudança intencional / padrão da casa**
   - Mudança de comportamento que claramente foi de propósito. Estilo que segue o padrão que
     já existe no resto do projeto (mesmo que você faria diferente).
   - Se há um `CLAUDE.md`/manual do projeto, ele manda no estilo. Não invente regra de estilo
     que não está lá.

6. **Tem um motivo válido (leia o contexto)**
   - Código que "parece" errado mas tem um comentário explicando, ou um motivo claro pelo
     arquivo em volta. Leia o entorno antes de acusar.

## Como decidir se um achado "passa" (a régua de confiança)

Para cada achado, estime sua **confiança** de que é um problema real e novo:

| Confiança | O que é | Rápida | Profunda |
|-----------|---------|--------|----------|
| baixa | provável falso-positivo, ou pré-existente, ou nitpick | corta | corta |
| média | real mas pequeno / caso raro | corta | inclui como 🟡 "vale melhorar" |
| alta | vai acontecer, atrapalha o usuário | inclui como 🟠 importante | inclui como 🟠 importante |
| altíssima | certo de que quebra / é buraco de segurança | inclui como 🔴 crítico | inclui como 🔴 crítico |

Regras de bolso pra calibrar a confiança:
- **É novo (está no diff)?** puxa a confiança pra cima. Pré-existente puxa pra baixo (quase
  sempre corta).
- **Causa falha de verdade** (quebra, perde dado, vaza segurança)? puxa forte pra cima.
- **É só estética/preferência?** puxa pra baixo — vira 🟢 boa prática ou nem entra.
- **Segredo exposto e injeção** entram sempre que forem reais, mesmo "pequenos" — o risco é
  alto demais pra filtrar.

## Mostre o que você filtrou (transparência gera confiança)

No parecer, sempre diga quantos sinais você **achou**, quantos **reportou** e quantos
**filtrou**. Isso mostra que você olhou tudo e escolheu com critério — e não que passou por
cima. Ex.: *"Achei 14 sinais na varredura; reportei 4 que importam e filtrei 10 (pré-existentes,
estilo e coisa que o editor já pega)."*

## Nunca invente

Se falta contexto pra ter certeza (você não viu a função chamada, não sabe de onde vem o
dado), **não afirme que é bug** — diga: *"a verificar: isto pode ser um problema se `X`; pra
confirmar, eu precisaria ver `Y`."* Um "a verificar" honesto vale mais que um alarme falso
confiante. E nunca cite um `arquivo:linha` ou uma função que você não viu de fato no diff.
