# Use Node 16 alpine as parent image
FROM node:16-alpine

# Change the working directory on the Docker image to /app
WORKDIR /face

# Copy the rest of the project files into this image
COPY . .


# Copy package.json and package-lock.json to the /app directory
COPY package.json package-lock.json ./

# Copy the environment file (adjust the filename if needed)
COPY .env ./

# docker running port 
ENV PORT=3001

# Install dependencies
RUN npm install

# Start the application
CMD npm start


