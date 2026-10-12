# Session lifecycle

```json
{
  "status": "needs_operator_recovery",
  "remote_directory": "/tmp/fleet-fk-f085448b233a45d5ba8ada5ad271265f",
  "host": "root@165.245.132.182",
  "session_id": "f085448b233a45d5ba8ada5ad271265f",
  "error": "Stop: export: deadline reached; elapsed=1192.1s; pid=48111",
  "recovery_commands": [
    [
      "scp",
      "-r",
      "root@165.245.132.182:/tmp/fleet-fk-f085448b233a45d5ba8ada5ad271265f",
      "/Users/ckl/code/fleet-mi300x/results/fk-20261012T135305Z/recovered-53a23698419c4302ba41d769cfaade07"
    ],
    [
      "/Users/ckl/code/fleet-mi300x/.venv/bin/python",
      "-u",
      "/Users/ckl/code/fleet-mi300x/tests/run_fk_session.py",
      "--verify-export",
      "--out",
      "/Users/ckl/code/fleet-mi300x/results/fk-20261012T135305Z/recovered-53a23698419c4302ba41d769cfaade07",
      "--session-id",
      "f085448b233a45d5ba8ada5ad271265f",
      "--expected-commit",
      "ce32cd515f705037646cb2db583b788729d76ac1"
    ],
    [
      "bash",
      "/Users/ckl/.hermes/cache/scratch/do_delete.sh",
      "608406276"
    ]
  ]
}
```
