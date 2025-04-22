# Use Node.js LTS (Latest LTS version)
FROM node:20-alpine

# Copy package files
COPY package.json package-lock.json* ./

COPY . ./

ARG POSTGRES_USER
ARG POSTGRES_PASSWORD
ARG POSTGRES_DB

RUN echo "POSTGRES_USER=$POSTGRES_USER" >> .env
RUN echo "POSTGRES_PASSWORD=$POSTGRES_PASSWORD" >> .env
RUN echo "POSTGRES_DB=$POSTGRES_DB" >> .env

# Install dependencies and PNPM
RUN npm install -g pnpm
RUN pnpm install

# Build the Next.js application
# RUN npm run build

# Expose port 3000
EXPOSE 3000

# Start the application
CMD ["npm", "run", "dev"]