# Quick Start: Claude Environment Setup

## First Time Setup

1. **Create your `.env` file with credentials:**
   ```bash
   # .env already exists with your credentials
   # If you need to update it, edit /home/v-murongma/code/OpenHands_SWE-Bench-Optimized/.env
   ```

2. **Load environment variables:**
   ```bash
   source setup_claude_env.sh
   ```

3. **Start the proxy server:**
   ```bash
   ./start_claude_proxy.sh
   ```

## Every Time You Start a New Shell

Just run:
```bash
cd /home/v-murongma/code/OpenHands_SWE-Bench-Optimized
source setup_claude_env.sh
./start_claude_proxy.sh
```

## Or Use One-Liner

```bash
cd /home/v-murongma/code/OpenHands_SWE-Bench-Optimized && source setup_claude_env.sh && ./start_claude_proxy.sh
```

## Files Created

- **`.env`** - Your actual credentials (gitignored, NEVER commit!)
- **`.env.example`** - Template showing what variables are needed
- **`setup_claude_env.sh`** - Helper script to load environment variables
- **`start_claude_proxy.sh`** - Script to start the proxy server

## Security Notes

- ✅ `.env` is in `.gitignore` - your credentials are safe
- ✅ `.env.claude` is in `.gitignore` too
- ✅ All hardcoded keys have been removed from tracked files
- ❌ **NEVER** commit `.env` or any file with actual API keys

## Verification

Check that `.env` is not tracked:
```bash
git status --short | grep "\.env"
# Should show nothing or "??" (untracked)
```

Check that environment is loaded:
```bash
echo $CLAUDE_API_KEY
# Should show your API key
```
