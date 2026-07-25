# Image de base Python légère
FROM python:3.11-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Installer les dépendances système
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY app/     ./app/
COPY config/  ./config/ 
COPY models/  ./models/
COPY certs/ ./certs/

# Lancer le pipeline
CMD ["python", "-m", "app.main"]