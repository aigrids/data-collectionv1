# Contribution guidelines for data-collectionv1

## Table of Contents
- [Getting Started](#-getting-started)
- [Keeping Your Branch Up to Date](#-keeping-your-branch-up-to-date)
- [Important Note on Rebasing](#️-important-note-on-rebasing)

We use a single long-lived branch for this project:

```cpp
----------------------------- main (default branch) -----------------------------
```

All feature and personal development branches should originate from `main`.


## Getting Started

To begin contributing:

1. Clone the repository
```bash
git clone https://github.com/aigrids/data-collectionv1

cd data-collectionv1
```

2. Create and switch to your personal development branch for making changes.
```bash
git checkout -b <your_personal_branch>
```

```lua
-------------------------------------main-------------------------------------
            ----------------<your_personal_branch>----------------
```

3. Create virtual enviornment:
```bash
python3.12 -m venv .venv

source .venv/bin/activate 
```

4. Install required packages in virtual enviornment:
```bash
python -m pip install -r requirements.txt
```

4. (Optional) Download all AI.grids data to root repo specified in config.yml:
 ```bash
python3 scripts/download.py
```

## Keeping Your Branch Up to Date

If multiple contributors are working simultaneously, it’s important to sync your 
branch with the latest changes from `main` before opening a pull request:

1. Ensure you're on your branch:
```bash
git checkout <your_personal_branch>
```

2. Fetch the latest changes and update `main`:
```bash
git fetch origin
git checkout main
git pull origin main
```

3. Merge `main` into your branch:
```bash
git checkout <your_personal_branch>
git merge main
```
This helps reduce merge conflicts during pull requests and ensures your work is 
based on the latest codebase. Resolve any conflicts that occur.

4. Push your updated branch to GitHub:
```bash
git push -u origin <your_personal_branch>
```

5. Open a pull request from `<your_personal_branch>` into `main` on GitHub.


## Important Note on Rebasing

* Allowed: Use git rebase locally to tidy up your commits before merging into a 
shared branch.
* Avoid: Rebasing commits that have already been pushed to the remote repository. 
This can cause issues for other collaborators.