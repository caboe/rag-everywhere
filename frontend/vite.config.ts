import tailwindcss from '@tailwindcss/vite'
import { sveltekit } from '@sveltejs/kit/vite'
import { defineConfig } from 'vite'

export default defineConfig({
	plugins: [tailwindcss(), sveltekit()],
	server: {
		// Add server configuration
		host: '0.0.0.0', // Allow connections from outside the container
		port: 5173, // Explicitly define the port (matches docker-compose)
		proxy: {
			// Proxy /api requests to the backend service
			'/api': {
				target: 'http://backend:8000', // Docker service name and port
				changeOrigin: true, // Recommended for virtual hosts
				// rewrite: (path) => path.replace(/^\/api/, ''), // Keep /api prefix for backend routes
			},
		},
	},
})
