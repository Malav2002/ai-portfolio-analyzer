FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Install additional dependencies for real data
RUN npm install yahoo-finance2 dotenv

# Copy source code
COPY . .

# Create routes directory if it doesn't exist
RUN mkdir -p routes

# Expose port
EXPOSE 3001

# Start the application
CMD ["npm", "start"]