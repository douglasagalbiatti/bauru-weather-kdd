# Coletor IPMet (timer do systemd)

O site do IPMet só responde a IPs brasileiros, então a coleta roda numa máquina no Brasil,
num timer de usuário do systemd (a cada 30 min, xx:07 e xx:37), e faz push para o GitHub.

Instalação (requer chave SSH sem senha com acesso de escrita ao repositório):

```bash
mkdir -p ~/coletor-ipmet && cd ~/coletor-ipmet
git clone git@github.com:douglasagalbiatti/bauru-weather-kdd.git
python3 -m venv bauru-weather-kdd/.venv && bauru-weather-kdd/.venv/bin/pip install requests
cp bauru-weather-kdd/deploy/coletor-ipmet/coletar.sh . && chmod +x coletar.sh
cp bauru-weather-kdd/deploy/coletor-ipmet/coletor-ipmet.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now coletor-ipmet.timer
loginctl enable-linger "$USER"
```

Acompanhar: `journalctl --user -u coletor-ipmet -n 50` e `systemctl --user list-timers`.
