<!-- frontend/src/routes/+page.svelte -->
<script lang="ts">
	import { onMount } from 'svelte'

	let stats: { total_documents: number; total_folders: number } | null = null
	let isLoading = true
	let error: string | null = null

	async function fetchStats() {
		isLoading = true
		error = null
		try {
			const response = await fetch('/api/dashboard/stats')
			if (!response.ok) {
				let errorDetail = 'Failed to fetch dashboard stats.'
				try {
					const errorData = await response.json()
					errorDetail = errorData.detail || errorDetail
				} catch (jsonError) {
					/* Ignore */
				}
				throw new Error(`HTTP error ${response.status}: ${errorDetail}`)
			}
			stats = await response.json()
		} catch (err: any) {
			console.error('Error fetching stats:', err)
			error = err.message || 'Could not fetch statistics.'
			stats = null
		} finally {
			isLoading = false
		}
	}

	onMount(() => {
		fetchStats()
	})
</script>

<div class="container mx-auto p-10 space-y-6">
	<h1 class="text-3xl font-bold text-center">Local RAG Dashboard</h1>

	{#if isLoading}
		<p class="text-center text-gray-500">Loading statistics...</p>
	{:else if error}
		<div
			class="p-4 rounded-md text-sm bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-200 text-center"
		>
			Error loading statistics: {error}
		</div>
	{:else if stats}
		<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
			<div
				class="p-4 border rounded-lg shadow bg-white dark:bg-gray-800 text-center"
			>
				<h2 class="text-xl font-semibold mb-2">Total Documents</h2>
				<p class="text-4xl font-bold">{stats.total_documents}</p>
			</div>
			<div
				class="p-4 border rounded-lg shadow bg-white dark:bg-gray-800 text-center"
			>
				<h2 class="text-xl font-semibold mb-2">Total Folders</h2>
				<p class="text-4xl font-bold">{stats.total_folders}</p>
			</div>
			<!-- Add more stat cards here later -->
		</div>
	{:else}
		<p class="text-center text-gray-500">Could not load statistics.</p>
	{/if}

	<!-- Optional: Add links or other dashboard elements -->
	<div class="text-center mt-6">
		<a
			href="/upload"
			class="btn btn-primary mr-2">Upload Documents</a
		>
		<a
			href="/chat"
			class="btn btn-secondary">Chat with Documents</a
		>
	</div>
</div>
