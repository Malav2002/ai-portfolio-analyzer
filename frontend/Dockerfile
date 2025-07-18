# frontend/Dockerfile - Fixed for lock file sync
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package.json first
COPY package.json ./

# Remove package-lock.json to avoid sync issues and regenerate it
RUN rm -f package-lock.json

# Install dependencies (this will create a new package-lock.json)
RUN npm install

# Copy source code
COPY . .

# Create .next directory
RUN mkdir -p .next

# Set environment variable for build
ENV NEXT_TELEMETRY_DISABLED=1

# Build the application
RUN npm run build

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]