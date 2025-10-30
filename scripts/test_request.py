import argparse, requests, json

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://localhost:8080/predict')
parser.add_argument('--image', required=False, help='Path to a CXR image')
parser.add_argument('--threshold', type=float, default=0.25)
args = parser.parse_args()

if not args.image:
    print('Usage: python scripts/test_request.py --image path/to/cxr.jpg --threshold 0.25')
    raise SystemExit(1)

with open(args.image, 'rb') as f:
    files = {'file': (args.image, f, 'image/jpeg')}
    r = requests.post(args.url, files=files, params={'threshold': args.threshold}, timeout=60)
print('Status:', r.status_code)
try:
    print(json.dumps(r.json(), indent=2))
except Exception:
    print(r.text)
