# Medical Image Data (via Kaggle)

## 1. Prerequisites

1. **Kaggle Account**
    - Sign up at https://www.kaggle.com/ if you don’t already have an account.



2. **Kaggle API Credentials**
    - Go to your Kaggle profile → **Account** → scroll to “API” → click “Create New API Token.”
    - This will download a file named `kaggle.json`.
    - Place `kaggle.json` in your home directory under `~/.kaggle/kaggle.json` (Linux/macOS) or `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
    - Make sure it has permissions `600` (Linux/macOS) or is otherwise only readable by you.

   
3. **Install Python Dependencies**
   ```bash
   pip install kaggle requests
