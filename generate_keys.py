# from pywebpush import Vapid
# import base64

# v = Vapid()
# v.generate_keys()

# # Export public key as Base64url string
# public_key_bytes = v.public_key.public_bytes(
#     encoding=__import__('cryptography.hazmat.primitives.serialization', fromlist=['Encoding']).Encoding.X962,
#     format=__import__('cryptography.hazmat.primitives.serialization', fromlist=['PublicFormat']).PublicFormat.UncompressedPoint
# )
# public_key_b64 = base64.urlsafe_b64encode(public_key_bytes).rstrip(b'=').decode('utf-8')

# # Export private key as Base64url string
# private_key_bytes = v.private_key.private_bytes(
#     encoding=__import__('cryptography.hazmat.primitives.serialization', fromlist=['Encoding']).Encoding.PEM,
#     format=__import__('cryptography.hazmat.primitives.serialization', fromlist=['PrivateFormat']).PrivateFormat.TraditionalOpenSSL,
#     encryption_algorithm=__import__('cryptography.hazmat.primitives.serialization', fromlist=['NoEncryption']).NoEncryption()
# )
# private_key_b64 = base64.urlsafe_b64encode(private_key_bytes).rstrip(b'=').decode('utf-8')

# print("=" * 60)
# print("VAPID_PUBLIC_KEY =", public_key_b64)
# print("VAPID_PRIVATE_KEY =", private_key_b64)
# print("=" * 60)


from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
import base64, os

# Generate EC key pair (P-256 curve for VAPID)
private_key = ec.generate_private_key(ec.SECP256R1())
public_key = private_key.public_key()

# Public key: uncompressed point format → base64url
pub_bytes = public_key.public_bytes(
    encoding=serialization.Encoding.X962,
    format=serialization.PublicFormat.UncompressedPoint
)
pub_b64 = base64.urlsafe_b64encode(pub_bytes).rstrip(b'=').decode()

# Private key: raw 32 bytes → base64url
priv_bytes = private_key.private_numbers().private_value.to_bytes(32, 'big')
priv_b64 = base64.urlsafe_b64encode(priv_bytes).rstrip(b'=').decode()

print("=" * 70)
print("Copy these to your Render.com environment variables:")
print("=" * 70)
print(f"\nVAPID_PUBLIC_KEY  = {pub_b64}")
print(f"\nVAPID_PRIVATE_KEY = {priv_b64}")
print("\n" + "=" * 70)
print(f"\nPublic key length: {len(pub_b64)} chars (should be ~87)")
print(f"Private key length: {len(priv_b64)} chars (should be ~43)")