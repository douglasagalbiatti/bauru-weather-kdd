# O parecer e o veredito do Crivo

Este é o formato de saída de uma revisão. O objetivo é: **claro, curto, honesto e
acionável** — um dono de negócio lê e entende, um programador lê e conserta. Sempre em
português simples.

## Estrutura do parecer

```
## Revisão do Crivo — <o que foi revisado>

**Resumo:** revisei <N> arquivo(s), <M> linha(s) alterada(s). Risco geral: <Baixo | Médio | Alto>.

| Sinais achados | Reportados | Filtrados |
|----------------|------------|-----------|
| <total>        | <reportados> | <filtrados> |

---

### 🔴 Crítico — não pode subir assim (<n>)
1. **<arquivo>:<linha>** — <o problema em uma frase>
   - **Por que importa:** <o risco na prática, em português de gente>
   - **Correção:**
     ```diff
     - <linha como está (errada)>
     + <linha como deveria ficar>
     ```

### 🟠 Importante — conserte antes de subir (<n>)
1. **<arquivo>:<linha>** — <problema>
   - **Por que importa:** <...>
   - **Correção:** <diff ou descrição objetiva>

### 🟡 Vale melhorar — não bloqueia (<n>)
1. **<arquivo>:<linha>** — <sugestão> — <ganho de fazer>

### 🟢 O que ficou bom
- <algo que a pessoa fez certo — sempre reconheça pelo menos uma coisa>

---

### Veredito: <✅ PODE SUBIR | 🟡 SUBA COM RESSALVAS | 🔴 NÃO SUBA AINDA>
<uma frase explicando o veredito>

**Antes de subir, resolva (na ordem):**
1. <o conserto que derruba mais risco>
2. <o segundo>
3. <o terceiro, se houver>
```

## Como escolher o veredito

- **🔴 NÃO SUBA AINDA** — existe pelo menos **um achado 🔴 Crítico**: algo que quebra pro
  usuário, perde/corrompe dado, ou é um buraco de segurança real (segredo exposto, injeção,
  XSS explorável, rota sem autorização). Não sobe até resolver.
- **🟡 SUBA COM RESSALVAS** — nada crítico, mas há achados **🟠 Importantes** que seria
  imprudente ignorar. Pode subir se for consciente/urgente, mas o certo é consertar primeiro.
  Deixe claro o que fica pendente.
- **✅ PODE SUBIR** — só 🟡/🟢 (melhorias e elogios), nada que atrapalhe o usuário ou a
  segurança. Suba tranquilo; as sugestões ficam pra quando der.

Seja **honesto no veredito** — este é o coração do Crivo. Não amoleça um 🔴 pra agradar, nem
exagere um 🟡 pra parecer rigoroso. O dono confia nessa decisão pra apertar (ou não) o botão
de subir.

## Regras de escrita do parecer

1. **Cada achado tem os três elementos:** onde (`arquivo:linha`), o quê (problema em uma
   frase) e **por que importa** (o risco concreto). Sem os três, não reporte.
2. **A correção é concreta.** Prefira o formato `diff` (linha `-` errada, linha `+` certa).
   Se não der pra mostrar em diff, descreva o passo exato ("troque a concatenação por consulta
   parametrizada: `cursor.execute('... WHERE id = ?', [id])`").
3. **Ordene por gravidade**, não pela ordem do arquivo. O 🔴 vem primeiro sempre.
4. **Elogie o que está certo (🟢).** Reconhecer o que ficou bom faz a pessoa confiar no resto
   e ensina o padrão bom. Sempre ache pelo menos uma coisa boa (se houver de verdade).
5. **Sem jargão solto.** Se usar um termo técnico, explique em meia frase o que é e o que dá
   de errado. O Crivo fala com o dono, não só com o dev.
6. **Curto.** Um parecer de 4 achados que importam vale mais que 20 observações. Se filtrou
   muita coisa, diga o número — não liste o ruído.
7. **Feche sempre com o veredito e os 1–3 consertos prioritários.** A pessoa tem que sair da
   revisão sabendo exatamente o que fazer agora.

## Exemplo curto (bom)

```
## Revisão do Crivo — mudanças não commitadas

**Resumo:** revisei 2 arquivos, 18 linhas alteradas. Risco geral: Alto.

| Sinais achados | Reportados | Filtrados |
|----------------|------------|-----------|
| 9              | 3          | 6         |

### 🔴 Crítico — não pode subir assim (2)
1. **app.py:12** — a busca no banco junta o texto com o que o usuário digita.
   - **Por que importa:** é injeção de SQL — alguém pode digitar um "comando" no lugar do
     id e ler (ou apagar) a tabela inteira de clientes.
   - **Correção:**
     ```diff
     - query = "SELECT * FROM users WHERE id = " + request.params["id"]
     + query = "SELECT * FROM users WHERE id = ?"
     + cursor.execute(query, [request.params["id"]])
     ```
2. **app.py:3** — tem uma chave de API escrita direto no código.
   - **Por que importa:** quem tiver acesso ao código tem a sua chave. Se isso já foi pro
     Git, considere a chave vazada — troque ela no serviço e leia de variável de ambiente.
   - **Correção:** `API_KEY = os.environ["API_KEY"]` e ponha a chave no `.env` (fora do Git).

### 🟡 Vale melhorar (1)
1. **ui.js:2** — comparação com `==`; use `===` pra evitar surpresa (0 == '' vira verdadeiro).

### 🟢 O que ficou bom
- A separação da função de envio ficou clara e com bom nome.

### Veredito: 🔴 NÃO SUBA AINDA
Tem injeção de SQL e uma chave exposta — os dois são risco real de segurança.

**Antes de subir, resolva (na ordem):**
1. Trocar a busca por consulta parametrizada (app.py:12).
2. Tirar a chave do código e revogá-la no serviço (app.py:3).
```
