"""
Generador y Administrador Autónomo de PKI local con extensiones SAN para mTLS.
Incluye KeyUsage, ExtendedKeyUsage, SubjectKeyIdentifier y AuthorityKeyIdentifier
conforme a RFC 5280 y OpenSSL 3.2+ / Python 3.14+.
"""
import os
import datetime
import ipaddress
from pathlib import Path
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class CertManager:
    def __init__(self, cert_dir: str = None):
        self.cert_dir = Path(cert_dir) if cert_dir else (PROJECT_ROOT / "certs")
        self.cert_dir.mkdir(parents=True, exist_ok=True)
        self.ca_key_path = str(self.cert_dir / "ca.key")
        self.ca_cert_path = str(self.cert_dir / "ca.crt")
        self.server_key_path = str(self.cert_dir / "server.key")
        self.server_cert_path = str(self.cert_dir / "server.crt")
        self.client_key_path = str(self.cert_dir / "client.key")
        self.client_cert_path = str(self.cert_dir / "client.crt")

    def _generate_rsa_key(self):
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )

    def _save_key(self, key, path: str):
        with open(path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

    def _save_cert(self, cert, path: str):
        with open(path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

    def generate_all_certs(self, overwrite: bool = False):
        """Genera CA raíz, certificado de servidor con SAN y certificado de cliente mTLS."""
        if not overwrite and all(os.path.exists(p) for p in [
            self.ca_cert_path, self.ca_key_path,
            self.server_cert_path, self.server_key_path,
            self.client_cert_path, self.client_key_path
        ]):
            return {
                "ca_cert": self.ca_cert_path,
                "server_cert": self.server_cert_path,
                "server_key": self.server_key_path,
                "client_cert": self.client_cert_path,
                "client_key": self.client_key_path
            }

        # 1. Generar CA Raíz
        ca_key = self._generate_rsa_key()
        ca_subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "SO"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Fenix Sovereign Cluster"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Fenix Root CA"),
        ])
        ca_ski = x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key())
        ca_aki = x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key())
        ca_key_usage = x509.KeyUsage(
            digital_signature=True,
            content_commitment=False,
            key_encipherment=False,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=True,
            crl_sign=True,
            encipher_only=False,
            decipher_only=False
        )

        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_subject)
            .issuer_name(ca_subject)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .add_extension(ca_key_usage, critical=True)
            .add_extension(ca_ski, critical=False)
            .add_extension(ca_aki, critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        self._save_key(ca_key, self.ca_key_path)
        self._save_cert(ca_cert, self.ca_cert_path)

        # 2. Generar Certificado de Servidor con SAN (IP + DNS)
        server_key = self._generate_rsa_key()
        server_subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "SO"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Fenix Sovereign Cluster"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])
        san_server = x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            x509.IPAddress(ipaddress.IPv6Address("::1")),
        ])
        server_ski = x509.SubjectKeyIdentifier.from_public_key(server_key.public_key())
        server_aki = x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key())
        server_key_usage = x509.KeyUsage(
            digital_signature=True,
            content_commitment=False,
            key_encipherment=True,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=False,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False
        )

        server_cert = (
            x509.CertificateBuilder()
            .subject_name(server_subject)
            .issuer_name(ca_subject)
            .public_key(server_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(server_key_usage, critical=True)
            .add_extension(san_server, critical=False)
            .add_extension(server_ski, critical=False)
            .add_extension(server_aki, critical=False)
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                critical=False
            )
            .sign(ca_key, hashes.SHA256())
        )
        self._save_key(server_key, self.server_key_path)
        self._save_cert(server_cert, self.server_cert_path)

        # 3. Generar Certificado de Cliente (Sargento mTLS)
        client_key = self._generate_rsa_key()
        client_subject = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "SO"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Fenix Sovereign Cluster"),
            x509.NameAttribute(NameOID.COMMON_NAME, "fenix-client-sargento"),
        ])
        san_client = x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
        ])
        client_ski = x509.SubjectKeyIdentifier.from_public_key(client_key.public_key())
        client_aki = x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key())
        client_key_usage = x509.KeyUsage(
            digital_signature=True,
            content_commitment=False,
            key_encipherment=True,
            data_encipherment=False,
            key_agreement=False,
            key_cert_sign=False,
            crl_sign=False,
            encipher_only=False,
            decipher_only=False
        )

        client_cert = (
            x509.CertificateBuilder()
            .subject_name(client_subject)
            .issuer_name(ca_subject)
            .public_key(client_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(client_key_usage, critical=True)
            .add_extension(san_client, critical=False)
            .add_extension(client_ski, critical=False)
            .add_extension(client_aki, critical=False)
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
                critical=False
            )
            .sign(ca_key, hashes.SHA256())
        )
        self._save_key(client_key, self.client_key_path)
        self._save_cert(client_cert, self.client_cert_path)

        return {
            "ca_cert": self.ca_cert_path,
            "server_cert": self.server_cert_path,
            "server_key": self.server_key_path,
            "client_cert": self.client_cert_path,
            "client_key": self.client_key_path
        }

if __name__ == "__main__":
    cm = CertManager()
    res = cm.generate_all_certs(overwrite=True)
    print("PKI regenerada conforme a RFC 5280 / OpenSSL 3.2+:")
    for k, v in res.items():
        print(f"  {k}: {v}")
