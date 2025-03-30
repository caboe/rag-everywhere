<!-- frontend/src/routes/upload/+page.svelte -->
<script lang="ts">
	// Skeleton imports removed
	// import { ProgressRadial } from '@skeletonlabs/skeleton';
	// import { toastStore } from '@skeletonlabs/skeleton';
	// import type { ToastSettings } from '@skeletonlabs/skeleton';

	let files: FileList | null = null
	let isLoading = false
	let uploadProgress = 0 // Placeholder for potential progress tracking
	let message: { type: 'success' | 'error' | 'warning'; text: string } | null =
		null // Simple message state

	async function handleUpload() {
		if (!files || files.length === 0) {
			// Use simple message state instead of toast
			message = {
				type: 'warning',
				text: 'Please select one or more files to upload.',
			}
			return
		}

		isLoading = true
		uploadProgress = 0 // Reset progress
		message = null // Clear previous messages

		const formData = new FormData()
		for (let i = 0; i < files.length; i++) {
			formData.append('files', files[i])
		}
		// TODO: Add parent_id selection logic later
		// formData.append('parent_id', selectedParentId || '');

		try {
			// TODO: Use environment variable for backend URL
			const response = await fetch(
				'http://localhost:8000/api/documents/upload',
				{
					method: 'POST',
					body: formData,
					// TODO: Add progress tracking using XMLHttpRequest if needed
				}
			)

			const result = await response.json()

			if (!response.ok) {
				throw new Error(
					result.detail || `HTTP error! status: ${response.status}`
				)
			}

			// Use simple message state instead of toast
			message = {
				type: 'success',
				text: `Successfully uploaded ${result.document_ids?.length || 0} file(s).`,
			}
			// Optionally clear the file input
			const fileInput = document.getElementById('fileInput') as HTMLInputElement
			if (fileInput) fileInput.value = ''
			files = null
		} catch (error: any) {
			console.error('Upload error:', error)
			// Use simple message state instead of toast
			message = {
				type: 'error',
				text: `Upload failed: ${error.message || 'Unknown error'}`,
			}
		} finally {
			isLoading = false
		}
	}
</script>

<div class="container mx-auto p-10 space-y-4">
	<h1 class="text-2xl font-bold mb-4">Upload Documents</h1>

	<!-- Simple Message Display -->
	{#if message}
		<div
			class="p-4 rounded-md text-sm"
			class:bg-green-100={message.type === 'success'}
			class:text-green-700={message.type === 'success'}
			class:dark:bg-green-900={message.type === 'success'}
			class:dark:text-green-200={message.type === 'success'}
			class:bg-red-100={message.type === 'error'}
			class:text-red-700={message.type === 'error'}
			class:dark:bg-red-900={message.type === 'error'}
			class:dark:text-red-200={message.type === 'error'}
			class:bg-yellow-100={message.type === 'warning'}
			class:text-yellow-700={message.type === 'warning'}
			class:dark:bg-yellow-900={message.type === 'warning'}
			class:dark:text-yellow-200={message.type === 'warning'}
		>
			{message.text}
		</div>
	{/if}

	<div
		class="p-4 space-y-4 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 shadow-md"
	>
		<div>
			<label
				class="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
				for="fileInput"
			>
				<span>Select Files</span>
				<input
					class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 dark:text-gray-400 focus:outline-none dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 p-2.5 mt-1 file:mr-4 file:py-2 file:px-4 file:rounded-md file:border-0 file:text-sm file:font-semibold file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100 dark:file:bg-primary-700 dark:file:text-primary-100 dark:hover:file:bg-primary-600"
					type="file"
					id="fileInput"
					multiple
					bind:files
					disabled={isLoading}
					accept=".pdf,.docx,.txt"
				/>
			</label>
			{#if files}
				<small class="text-xs text-gray-500 dark:text-gray-400 mt-1"
					>Selected {files.length} file(s).</small
				>
			{/if}
		</div>

		<!-- TODO: Add Tree View component here for selecting parent_id -->
		<div>
			<label
				class="block mb-2 text-sm font-medium text-gray-900 dark:text-white"
			>
				<span>Target Folder (TODO)</span>
				<input
					type="text"
					class="block w-full p-2.5 mt-1 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg cursor-not-allowed text-gray-500 dark:text-gray-400"
					value="Root (Not implemented)"
					disabled
				/>
			</label>
		</div>

		<button
			type="button"
			class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
			on:click={handleUpload}
			disabled={isLoading || !files || files.length === 0}
		>
			{#if isLoading}
				<!-- Simple loading indicator -->
				<svg
					class="animate-spin -ml-1 mr-3 h-5 w-5 text-white"
					xmlns="http://www.w3.org/2000/svg"
					fill="none"
					viewBox="0 0 24 24"
				>
					<circle
						class="opacity-25"
						cx="12"
						cy="12"
						r="10"
						stroke="currentColor"
						stroke-width="4"
					></circle>
					<path
						class="opacity-75"
						fill="currentColor"
						d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
					></path>
				</svg>
				<span>Uploading...</span>
			{:else}
				<span>Upload</span>
			{/if}
		</button>
	</div>

	<!-- TODO: Add Tree View display here -->
	<div
		class="p-4 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 shadow-md"
	>
		<h2 class="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
			Document Tree (TODO)
		</h2>
		<p class="text-sm text-gray-600 dark:text-gray-400">
			The document tree structure will be displayed here.
		</p>
	</div>
</div>
