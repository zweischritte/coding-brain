# Coding Brain Documentation

This documentation site uses [Docsify](https://docsify.js.org/) to dynamically render all markdown files in the project.

## Quick Start

### View Documentation (with Docker Compose)

The docs server starts automatically when you run:

```bash
cd openmemory && make up
```

Then open: <http://localhost:3080/docs/>

### View Documentation (standalone)

From the project root, run:

```bash
python -m http.server 3000
```

Then open: <http://localhost:3000/docs/>

### Update Sidebar

When you add, remove, or rename markdown files, regenerate the sidebar:

```bash
./scripts/generate-docs-sidebar.sh
```

## Structure

The sidebar is auto-generated and includes markdown files from:

- `/` - Project root (README, CONTRIBUTING, etc.)
- `/docs/` - Core documentation
- `/openmemory/` - OpenMemory component docs
- `/openmemory/api/app/guidance/` - Guidance documents
- `/embedchain/` - Embedchain docs and examples
- `/mem0-ts/` - TypeScript SDK docs
- And more...

## Files

| File                               | Purpose                     |
| ---------------------------------- | --------------------------- |
| `docs/index.html`                  | Docsify entry point         |
| `docs/_sidebar.md`                 | Auto-generated navigation   |
| `scripts/generate-docs-sidebar.sh` | Sidebar generation script   |

## Features

- Full-text search across all documentation
- Syntax highlighting for code blocks
- No build step required - edits are visible immediately on refresh
- Works with any static file server
