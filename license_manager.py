import os
import hashlib
import json
from datetime import datetime, timedelta

LICENSE_FILE = os.path.join(os.path.dirname(__file__), 'license.json')

PLANS = {
    'starter': {
        'cameras': 1,
        'cloud': False,
        'support': False,
        'price': 0,
        'name': 'Starter'
    },
    'pro': {
        'cameras': 999,
        'cloud': True,
        'support': True,
        'price': 49,
        'name': 'Pro'
    },
    'enterprise': {
        'cameras': 999,
        'cloud': True,
        'support': True,
        'api': True,
        'price': 0,
        'name': 'Enterprise'
    }
}


def generate_license_key(plan: str, email: str, expires_days: int = 365) -> str:
    """Génère une clé de licence simple (hash)."""
    base = f"{plan}:{email}:{datetime.now().strftime('%Y-%m-%d')}:visionagent"
    return hashlib.sha256(base.encode()).hexdigest()[:32].upper()


def load_license() -> dict:
    if not os.path.exists(LICENSE_FILE):
        return {'plan': 'starter', 'key': None, 'email': None, 'expires': None}
    try:
        with open(LICENSE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {'plan': 'starter', 'key': None, 'email': None, 'expires': None}


def save_license(license_info: dict):
    with open(LICENSE_FILE, 'w', encoding='utf-8') as f:
        json.dump(license_info, f, indent=2)


def activate_license(key: str, email: str) -> dict:
    """Active une licence. Accepte un format simple KEY:PLAN ou une clé seule."""
    plan = 'starter'
    if ':' in key:
        key, plan = key.split(':', 1)
    if plan not in PLANS:
        plan = 'starter'
    expected = generate_license_key(plan, email)
    if key == expected:
        license_info = {
            'plan': plan,
            'key': key,
            'email': email,
            'expires': (datetime.now() + timedelta(days=365)).isoformat()
        }
        save_license(license_info)
        return {'success': True, 'plan': plan}
    return {'success': False, 'error': 'Clé invalide'}


def get_plan() -> dict:
    lic = load_license()
    plan_id = lic.get('plan', 'starter')
    plan = PLANS.get(plan_id, PLANS['starter']).copy()
    plan['id'] = plan_id
    plan['expires'] = lic.get('expires')
    return plan


def check_limit(limit_name: str, value: int) -> bool:
    plan = get_plan()
    limit = plan.get(limit_name)
    if limit is None:
        return True
    return value <= limit


if __name__ == '__main__':
    # Exemple de génération de clé Pro
    email = "demo@client.com"
    plan = "pro"
    key = generate_license_key(plan, email)
    print(f"Plan: {plan}")
    print(f"Email: {email}")
    print(f"Clé: {key}")
    print(f"Format d'activation: {key}:{plan}")
