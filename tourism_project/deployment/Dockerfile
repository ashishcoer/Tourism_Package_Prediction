# Use a minimal base image with Python 3.9 installed.
FROM python:3.9

# Set the working directory inside the container to /app.
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory.
COPY . .

# Install Python dependencies listed in requirements.txt.
RUN pip3 install -r requirements.txt

# Create a non-root user 'user' with UID 1000 and set ownership.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Change working directory to the user's application directory.
WORKDIR $HOME/app

# Copy the application files to the user's directory.
COPY --chown=user . $HOME/app

# Define the command to run the Streamlit app on port "8501" and make it accessible externally.
# `--server.address=0.0.0.0` makes the app reachable from any IP address.
# `--server.enableXsrfProtection=false` disables CSRF protection, often needed for public deployments.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
