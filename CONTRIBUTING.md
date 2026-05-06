# Contribution guidelines for PowerSystemData

## Table of Contents
- [Getting Started](#-getting-started)

We use a single long-lived branch for this project:

```cpp
----------------------------- main (default branch) -----------------------------
```

All feature and personal development branches should originate from `main`.

## Getting Started
To begin contributing:
1. Clone the repository
```bash
git clone https://<your_personal_access_token>@github.com/ArsamAryandoust/PowerSystemData

cd PowerSystemData
```

2. Create and switch to your personal development branch for making changes.
```bash
git checkout -b <your_personal_branch>
```

```lua
-------------------------------------main-------------------------------------
            ----------------<your_personal_branch>----------------
```

3. Install required packages in virtual enviornment:
```bash
python3.12 -m venv .venv
source .venv/bin/activate 
python -m pip install -r requirements.txt
```

4. (Optional) Download all AI.grids data to root repo specified in config.yml:
 ```bash
python3 scripts/download.py
```