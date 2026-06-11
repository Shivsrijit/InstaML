FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

EXPOSE 5173

# Start Vite dev server on host 0.0.0.0
CMD ["npm", "run", "dev", "--", "--host"]
