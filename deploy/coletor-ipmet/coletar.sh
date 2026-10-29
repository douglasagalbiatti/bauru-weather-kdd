#!/usr/bin/env bash
# Coletor da estação IPMet/UNESP Bauru (substitui o agendamento do GitHub Actions).
# Executado pelo timer de usuário do systemd: ~/.config/systemd/user/coletor-ipmet.timer
#
# 1. git pull --rebase
# 2. src/update_ipmet.py (falha do site não interrompe a execução)
# 3. Se a leitura mudou: commit e push (até 3 tentativas, com pull --rebase entre elas)

set -uo pipefail

BASE="$HOME/coletor-ipmet"
REPO="$BASE/bauru-weather-kdd"
LOCK="$BASE/coletar.lock"
BRANCH="main"
ARQUIVOS=(data/interim/ipmet_latest.json data/raw/ipmet/)

# Nunca duas instâncias ao mesmo tempo
exec 9>"$LOCK"
if ! flock -n 9; then
    echo "Outra coleta já está em andamento; saindo."
    exit 0
fi

# Roda sem terminal: SSH nunca pode pedir senha nem confirmação
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519"
export GIT_TERMINAL_PROMPT=0

cd "$REPO" || { echo "Repositório não encontrado: $REPO"; exit 1; }

sincronizar() {
    if ! git pull --rebase --quiet origin "$BRANCH"; then
        echo "Aviso: git pull --rebase falhou."
        git rebase --abort 2>/dev/null || true
        return 1
    fi
}

sincronizar || true

if ! .venv/bin/python src/update_ipmet.py; then
    echo "Aviso: coleta do IPMet falhou nesta execução (site fora do ar ou leitura inválida)."
fi

if [ -n "$(git status --porcelain -- "${ARQUIVOS[@]}")" ]; then
    git add -- "${ARQUIVOS[@]}"
    git commit --quiet -m "chore(data): leitura da estacao IPMet Bauru [skip ci]"
    echo "Commit criado: $(git log --oneline -1)"
else
    echo "Nenhuma leitura nova a comitar."
fi

# Envia também commits que ficaram pendentes de execuções anteriores
if [ "$(git rev-list --count "origin/$BRANCH..HEAD" 2>/dev/null || echo 0)" -eq 0 ]; then
    exit 0
fi

for tentativa in 1 2 3; do
    if git push --quiet origin "HEAD:$BRANCH"; then
        echo "Push concluído (tentativa $tentativa)."
        exit 0
    fi
    echo "Push falhou (tentativa $tentativa); sincronizando e tentando de novo..."
    sleep 10
    sincronizar || true
done

echo "Falha ao enviar o commit após 3 tentativas."
exit 1
