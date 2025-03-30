<!-- frontend/src/routes/+page.svelte -->
<script lang="ts">
	// Page specific script can go here
	let count = 0
</script>

<div class="container h-full mx-auto flex justify-center items-center">
	<div class="space-y-5 text-center">
		<h1 class="h1">Welcome to Local RAG</h1>
		<p>This is the home page.</p>

		<button
			class="btn variant-filled-primary"
			on:click={() => count++}
		>
			Clicks: {count}
		</button>
	</div>
</div>
