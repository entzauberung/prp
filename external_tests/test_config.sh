#!/bin/bash
# Test configuration loading with real credentials

export PRP_EXTERNAL_TESTS=1
export PRP_EXTERNAL_PROFILE_ALIASES=DEEPSEEK_FLASH_CHAT

# DeepSeek configuration
export PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_BASE_URL="https://api.deepseek.com/anthropic"
export PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_API_KEY="${DEEPSEEK_API_KEY:-your-api-key-here}"
export PRP_EXTERNAL_DEEPSEEK_FLASH_CHAT_MODEL="deepseek-v4-flash"

# Test loading
uv run python -c "
from external_tests.support import load_external_config

try:
    config = load_external_config()
    print('✓ Config loaded successfully')
    print('✓ Allowed hosts:', config.allowed_hosts)
    print('✓ Number of profiles:', len(config.profiles))

    # Show profile details
    for profile in config.profiles:
        print(f'✓ Profile: {profile.alias}')
        print(f'  - Vendor: {profile.vendor}')
        print(f'  - Model ID: {profile.model_id}')
        print(f'  - Protocol: {profile.protocol}')
        host = profile.base_url.split('://')[1].split('/')[0]
        print(f'  - Base URL host: {host}')
except Exception as e:
    import traceback
    print('ERROR:', e)
    traceback.print_exc()
    exit(1)
"
