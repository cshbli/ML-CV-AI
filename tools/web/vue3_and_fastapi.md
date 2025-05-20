# Build a web application with Vue 3 as frontend and FastAPI as backend

- <b>Frontend</b>: A <b>Vue 3</b> application using <b>Vite</b> for fast development and <b>Tailwind CSS</b> for styling. 
- <b>Backend</b>: A <b>FastAPI</b> application.
- <b>Communication</b>: The frontend will make HTTP requests to the backend using <b>Axios</b>.

## Step 1: Set Up the Backend with FastAPI

1. Create a project directory:
```bash
mkdir todo-app
cd todo-app
mkdir backend
cd backend
```

2. Set up a Python virtual environment

3. Install FastAPI and dependencies:
```bash
pip install fastapi uvicorn pydantic
```

4. Create the FastAPI application:

5. run the FastAPI server:
```bash
uvicorn main:app --reload
```

## Step 2: Set Up the Frontend with Vue 3

1. Navigate to the project root and create the frontend directory:
```bash
mkdir frontend
cd frontend
```

2. Initialize a Vue 3 project with Vite:
```bash
npm create vite@latest . -- --template vue
npm install
```

3. Install dependencies:
    - Axios for HTTP requests
    - Tailwind CSS for styling
```    
npm install axios
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p    
```

We may got this error while running "npx tailwindcss init -p"
```
npm error could not determine executable to run
```
To fix it, we will need downgrade the tailwindcss version to "^3.4.17" in the `package.json` file. 

4. Configure Tailwind CSS:

Edit `tailwind.config.js`
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{vue,js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

Create `src/assets/main.css`
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

Update `src/main.js` to include the CSS:
```javascript
import { createApp } from 'vue'
import App from './App.vue'
import './assets/main.css'

createApp(App).mount('#app')
```

5. Create the Vue components:

- Replace `src/App.vue` with the main application logic.
- Create a `src/components/TodoList.vue` component for rendering todos.

6. Run the Vue development server:
```bash
npm run dev
```

## Step 3: Test the Application

1. Ensure both servers are running:
    - Backend: `uvicorn main:app --reload` in the backend directory.
    - Frontend: `npm run dev` in the frontend directory.

## Step 4: Additional Notes

- <b>CORS</b>: The backend includes CORS middleware to allow requests from the frontend (http://localhost:5173). Update the allow_origins list if you deploy to a different domain.
- <b>Data Persistence</b>: This example uses an in-memory list for todos. For a production app, replace it with a database (e.g., `SQLite`, `PostgreSQL`) using an ORM like `SQLAlchemy`.
- <b>Styling</b>: Tailwind CSS provides basic styling. Customize the design further as needed.
Deployment:
- <b>Backend</b>: Deploy FastAPI using a WSGI server like Gunicorn and a reverse proxy like Nginx. Host on platforms like Heroku, AWS, or DigitalOcean.
- <b>Frontend</b>: Build the Vue app (`npm run build`) and serve the static files using a web server (e.g., Nginx) or a platform like Vercel or Netlify.

