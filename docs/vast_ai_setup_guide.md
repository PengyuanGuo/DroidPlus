# Vast.ai training instance setup

Local machine (macOS or Linux) → SSH / Cursor → Vast.ai → GitHub, Claude Code, Zellij

Updated: 2026-10-24. **LOCAL** = your laptop/desktop (macOS or Linux); **VAST** = the Ubuntu/Debian container. Commands are identical on macOS and Linux unless shown otherwise.

## 1. Create the instance — Vast.ai website

1. Add credit and pick a GPU. For OpenPI full finetuning, start with A100 80GB or compare H100 quotes. Check **per-GPU** VRAM.
2. Check CPU/RAM, disk, network, reliability, and max rental duration. ~128 GB RAM and 200–300 GB disk are starting estimates — budget for dataset, dependencies, base weights, and checkpoints.
3. Choose **On-demand** and a CUDA/PyTorch template with **SSH** or **Jupyter + SSH**.
4. Set **Disk Space before renting** — it cannot be resized later.
5. Register your public key (section 2), then Rent. For an existing instance, attach the key via its SSH-key control.
6. From **Connect → SSH**, copy the username, public IP, and SSH port (the public port mapped to container port 22).

**Billing:** A running instance bills even at 0% GPU and after you disconnect. **Stop** ends compute billing but keeps disk charges and releases the GPU (restart depends on availability). **Destroy** deletes container data.

## 2. SSH keys — LOCAL, one-time

| Connection | Private key (local) | Public key registered with |
|---|---|---|
| Local → Vast | `~/.ssh/id_ed25519_vast` | Vast account / instance |
| Vast → GitHub (forwarded agent) | `~/.ssh/id_ed25519` | GitHub account |

Check existing keys, and generate the Vast key only if it does not exist (never overwrite):

```bash
ls ~/.ssh/*.pub
ssh-keygen -t ed25519 -C "vast" -f ~/.ssh/id_ed25519_vast
```

Copy the public key and paste it into Vast.ai **SSH Keys**:

```bash
# macOS
pbcopy < ~/.ssh/id_ed25519_vast.pub
# Linux (Wayland / X11) — or just `cat` it and copy manually
wl-copy < ~/.ssh/id_ed25519_vast.pub
xclip -selection clipboard < ~/.ssh/id_ed25519_vast.pub
```

Private keys never leave your local machine. New instances can reuse the same public key.

## 3. SSH config and Cursor — LOCAL

Add this block to `~/.ssh/config` (keep other hosts); replace HostName/Port for each new instance:

```sshconfig
Host vast-atomic
    HostName 216.129.245.167
    User root
    Port 41037
    IdentityFile ~/.ssh/id_ed25519_vast
    IdentitiesOnly yes
    ForwardAgent yes
    ServerAliveInterval 30
```

```bash
chmod 600 ~/.ssh/config
ssh vast-atomic
```

Verify the host fingerprint on first connect; do not disable host-key checking.

**Cursor:** install **Remote - SSH** → Command Palette (`Cmd+Shift+P` on macOS, `Ctrl+Shift+P` on Linux) → **Remote-SSH: Connect to Host...** → `vast-atomic` → Open Folder → `/workspace`.

Optional port forwarding for a remote web service: `ssh -L 8080:localhost:8080 vast-atomic`, then open `http://localhost:8080` locally.

## 4. GitHub agent forwarding — LOCAL first

Make sure an agent is running and load your GitHub key:

```bash
# Linux only, if `ssh-add -l` says "Could not open a connection to your authentication agent":
eval "$(ssh-agent -s)"

ssh-add ~/.ssh/id_ed25519           # macOS: add --apple-use-keychain to remember the passphrase
ssh-add -l
ssh -T git@github.com               # expect "Hi <user>! You've successfully authenticated" (exit code 1 is normal)
```

If the key is not on GitHub yet, copy `~/.ssh/id_ed25519.pub` (same clipboard commands as section 2) into GitHub **Settings → SSH and GPG keys**.

Test forwarding end-to-end:

```bash
ssh -A vast-atomic 'echo "SSH_AUTH_SOCK=$SSH_AUTH_SOCK"; ssh-add -l; ssh -T git@github.com'
```

Expected: a socket path, your key fingerprints, and the GitHub greeting. Enable forwarding only for trusted hosts (never under `Host *`) — anyone with access to the remote socket can use your agent while you are connected.

## 5. Prepare the container — VAST

Read the instance guide first (and point any AI agent to it):

```bash
cat /etc/vast-agents-guide.md
```

Inspect, then create a working directory:

```bash
nvidia-smi; df -h /workspace; free -h; uname -m
ssh-add -l && ssh -T git@github.com
mkdir -p /workspace/pengyuan && cd /workspace/pengyuan
```

The template may activate `/venv/main`; keep project dependencies in separate environments.

## 6. Clone, pull, push — VAST

```bash
cd /workspace/pengyuan
git clone git@github.com:purdue-mars/atomic_policy.git
git clone git@github.com:PengyuanGuo/DroidPlus.git
```

Existing clone:

```bash
cd /workspace/pengyuan/atomic_policy
git status && git remote -v && git pull --ff-only
# wrong remote? git remote set-url origin git@github.com:purdue-mars/atomic_policy.git
```

Set identity per repository, then commit only intended files:

```bash
git config user.name "Pengyuan Guo"
git config user.email "YOUR_GITHUB_EMAIL"

git diff
git add path/to/changed_file.py
git commit -m "Describe the change"
git push
```

Never commit keys, tokens, datasets, or checkpoints.

## 7. Troubleshooting

**`Permission denied (publickey)` logging into Vast** — on LOCAL, force the Vast key:

```bash
ssh -i ~/.ssh/id_ed25519_vast -o IdentitiesOnly=yes -p 41037 root@216.129.245.167
```

**Remote `ssh-add`: Could not open a connection to your authentication agent** — check `ssh-add -l` locally (on Linux, start the agent as in section 4), confirm `ForwardAgent yes`, reconnect. Don't start a separate empty agent on Vast.

**Works over plain SSH but not inside tmux/Zellij** — long-lived panes keep a stale `SSH_AUTH_SOCK`. Get the current value from a live SSH shell (`echo $SSH_AUTH_SOCK`, or `tmux show-environment SSH_AUTH_SOCK`), then in the pane:

```bash
export SSH_AUTH_SOCK=/tmp/ssh-XXXX/agent.NNNN   # current path only
ssh-add -l
```

Simplest: do Git in a fresh SSH shell outside the multiplexer.

**Agent has keys but GitHub refuses** — an old `Host github.com` block in the remote `~/.ssh/config` may force a deploy key. Test with `ssh -F /dev/null -T git@github.com`; if that works, fix that block.

## 8. Install Claude Code — VAST

```bash
apt-get update && apt-get install -y curl ca-certificates git tar unzip   # if missing
curl -fsSL https://claude.ai/install.sh | bash
grep -qxF 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"
claude --version
```

Run it in the project and follow the login prompts — on a headless server, open the printed URL in your local browser:

```bash
cd /workspace/pengyuan/atomic_policy
claude
```

Don't put credentials in tracked files; keep permission checks on.

## 9. Install Zellij — VAST

Zellij keeps sessions alive across SSH disconnects (not across Stop/Destroy). Install the prebuilt binary:

```bash
(
  set -eu
  case "$(uname -m)" in
    x86_64) t=x86_64-unknown-linux-musl ;;
    aarch64|arm64) t=aarch64-unknown-linux-musl ;;
    *) echo "Unsupported arch" >&2; exit 1 ;;
  esac
  d=$(mktemp -d); trap 'rm -rf "$d"' EXIT
  curl -fL "https://github.com/zellij-org/zellij/releases/latest/download/zellij-$t.tar.gz" | tar -xz -C "$d"
  install -Dm755 "$d/zellij" "$HOME/.local/bin/zellij"
)
zellij --version
```

**Avoid nesting in auto-tmux:** if the login banner mentions `.no_auto_tmux`, run `touch ~/.no_auto_tmux`, detach tmux (**Ctrl+B, D**), and reconnect. `echo "$TMUX"` should be empty before starting Zellij.

```bash
zellij attach --create atomic   # start or reattach
zellij list-sessions
```

Detach with **Ctrl+O, D** (default bindings). Don't quit the session if jobs should keep running.

## 10. Upload data — LOCAL

```bash
ssh vast-atomic 'mkdir -p /workspace/datasets'
scp /path/to/teleop_20261010.zip vast-atomic:/workspace/datasets/
# large/resumable transfers (rsync is preinstalled on most Linux; on macOS use Homebrew rsync if needed):
rsync -avP /path/to/teleop_20261010.zip vast-atomic:/workspace/datasets/
```

Example local paths: macOS `~/Desktop/...`, Linux `~/Downloads/...`; quote paths that contain spaces.

Extract on VAST without overwriting:

```bash
cd /workspace/datasets
df -h /workspace
unzip -n teleop_20261010.zip -d teleop_20261010
```

Raw teleop data may still need conversion before finetuning.

## 11. Daily workflow

**LOCAL:**

```bash
ssh-add ~/.ssh/id_ed25519     # Linux: start ssh-agent first if needed
ssh vast-atomic
```

**VAST:**

```bash
ssh -T git@github.com
cd /workspace/pengyuan/atomic_policy && git pull --ff-only
zellij attach --create atomic
```

Before **Stop**, save checkpoints. Before **Destroy**, download and verify anything valuable — stopped storage is not a backup.

**New instance checklist:**

- Authorize your existing public key; update the alias's HostName/Port.
- Read `/etc/vast-agents-guide.md`; verify GPU, disk, and forwarded GitHub auth.
- Install Claude Code, Zellij, and project dependencies.
- Clone code and transfer data/checkpoints (nothing carries over from old instances).

## References

- Vast instances: https://docs.vast.ai/guides/instances/choosing/find-and-rent
- Vast SSH: https://docs.vast.ai/guides/instances/connect/ssh
- Vast storage: https://docs.vast.ai/guides/instances/storage/types
- GitHub agent forwarding: https://docs.github.com/en/authentication/connecting-to-github-with-ssh/using-ssh-agent-forwarding
- Claude Code: https://code.claude.com/docs/en/setup
- Zellij: https://zellij.dev/documentation/installation.html
