<!-- frontend/src/routes/+layout.svelte -->
<script lang="ts">
	// Tailwind CSS is imported via app.postcss
	import '../app.css'

	// Skeleton components removed

	// Optional: Add any global layout logic or state here
</script>

<!-- Basic Layout Wrapper -->
<div
	class="flex flex-col min-h-screen bg-white dark:bg-gray-900 text-black dark:text-white"
>
	<!-- Header Placeholder -->
	<header
		class="p-4 bg-gray-100 dark:bg-gray-800 border-b border-gray-300 dark:border-gray-600"
	>
		<div class="container mx-auto flex justify-between items-center">
			<strong class="text-xl uppercase">Local RAG</strong>
			<nav class="space-x-4">
				<a
					href="/"
					class="hover:text-primary-500">Home</a
				>
				<a
					href="/upload"
					class="hover:text-primary-500">Upload</a
				>
				<a
					href="/chat"
					class="hover:text-primary-500">Chat</a
				>
			</nav>
			<a
				class="text-primary-500 hover:underline"
				href="https://github.com/your-repo"
				target="_blank"
				rel="noreferrer"
			>
				GitHub
			</a>
		</div>
	</header>

	<!-- Page Route Content -->
	<main class="flex-grow container mx-auto p-4">
		<slot />
	</main>

	<!-- Footer Placeholder -->
	<footer class="p-4 text-center text-sm text-gray-500 dark:text-gray-400">
		© {new Date().getFullYear()} Local RAG System
	</footer>
</div>
