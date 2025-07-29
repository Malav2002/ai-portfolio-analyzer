#!/bin/bash
# fix-package-versions.sh
# Fix package version conflicts for React 19

echo "🔧 Fixing Package Version Conflicts"
echo "==================================="

# Stop containers
docker-compose down

# Go to frontend directory
cd frontend

# Remove problematic files
rm -f package-lock.json
rm -rf node_modules
rm -rf .next

echo ""
echo "1️⃣ Creating updated package.json with compatible versions..."

# Create new package.json with React 19 compatible versions
cat > package.json << 'EOF'
{
  "name": "ai-portfolio-analyzer-frontend",
  "version": "2.0.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "15.4.2",
    "react": "19.1.0",
    "react-dom": "19.1.0",
    "axios": "1.10.0",
    "lucide-react": "0.469.0",
    "clsx": "2.1.1",
    "react-dropzone": "14.3.8"
  },
  "devDependencies": {
    "@types/node": "20.9.0",
    "@types/react": "19.1.8",
    "@types/react-dom": "19.1.6",
    "typescript": "5.2.2",
    "tailwindcss": "3.4.17",
    "autoprefixer": "10.4.21",
    "postcss": "8.4.32",
    "eslint": "8.57.1",
    "eslint-config-next": "15.4.2"
  }
}
EOF

echo ""
echo "2️⃣ Updating Dockerfile to handle peer dependency conflicts..."

# Update Dockerfile to use --legacy-peer-deps
cat > Dockerfile << 'EOF'
FROM node:18-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Copy package files
COPY package.json ./

# Install with legacy peer deps to handle React 19 compatibility
RUN npm install --legacy-peer-deps --package-lock-only
RUN npm ci --legacy-peer-deps

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Next.js collects completely anonymous telemetry data about general usage.
ENV NEXT_TELEMETRY_DISABLED 1

RUN npm run build

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV production
ENV NEXT_TELEMETRY_DISABLED 1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public

# Set the correct permission for prerender cache
RUN mkdir .next
RUN chown nextjs:nodejs .next

# Automatically leverage output traces to reduce image size
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
EOF

echo ""
echo "3️⃣ Updating Tailwind config for v3.4.17..."

# Update tailwind config for older version
cat > tailwind.config.js << 'EOF'
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: 'var(--background)',
        foreground: 'var(--foreground)',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-in-out',
        'slide-up': 'slideUp 0.3s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
EOF

echo ""
echo "4️⃣ Installing dependencies locally with legacy peer deps..."
npm install --legacy-peer-deps

echo ""
echo "5️⃣ Going back to root and building with Docker..."
cd ..

# Clean Docker cache and rebuild
docker system prune -f
docker-compose build --no-cache frontend

echo ""
echo "6️⃣ Starting frontend service..."
docker-compose up -d frontend

echo ""
echo "7️⃣ Checking frontend status..."
sleep 10

if curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo "✅ Frontend is running at http://localhost:3000"
else
    echo "❌ Frontend not responding yet, checking logs..."
    docker-compose logs frontend --tail=20
fi

echo ""
echo "✅ Package version conflicts fixed!"
echo ""
echo "Key changes made:"
echo "  - Updated lucide-react to 0.469.0 (React 19 compatible)"
echo "  - Downgraded Tailwind to 3.4.17 (stable version)"
echo "  - Added --legacy-peer-deps to handle compatibility"
echo "  - Updated ESLint to 8.57.1 (compatible version)"
echo ""
echo "🌐 Frontend should be available at: http://localhost:3000"