# Security Policy

Security is a first-class concern for this platform, especially because it
processes customer PII and integrates with CRM/ESP providers.

## Reporting a vulnerability

If you discover a security vulnerability, please report it privately rather
than opening a public issue.

Please include:

- A clear description of the vulnerability.
- Steps to reproduce it.
- The affected component or module.
- Any suggested remediation.

We will acknowledge receipt, investigate, and aim to provide an initial
assessment promptly.

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security best practices

- Never commit `.env`, credentials, or service-account keys. The repository
  provides [`.env.example`](.env.example) as a template.
- Keep PII in a dedicated vault and use the encryption key from the environment
  (see the KVKK/PII section of `.env.example`).
- Enable `pre-commit` locally; it includes `detect-private-key` to prevent
  accidental credential commits.
- Review dependency updates from Dependabot regularly.
