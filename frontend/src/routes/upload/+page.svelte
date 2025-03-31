<!-- frontend/src/routes/upload/+page.svelte -->
<script lang="ts">
	import { onMount } from 'svelte' // Import onMount
	import TreeNode from '$lib/components/TreeNode.svelte' // Import the new component

	// Skeleton imports removed
	// import { ProgressRadial } from '@skeletonlabs/skeleton';
	// import { toastStore } from '@skeletonlabs/skeleton';
	// import type { ToastSettings } from '@skeletonlabs/skeleton';

	// Upload state
	let files: FileList | null = null
	let isUploading = false // Renamed from isLoading for clarity
	let uploadProgress = 0 // Placeholder for potential progress tracking
	let uploadMessage: {
		type: 'success' | 'error' | 'warning'
		text: string
	} | null = null // Renamed from message

	// Tree state
	let documentTree: any[] = [] // To store the fetched tree data
	let isTreeLoading = false
	let treeError: string | null = null
	let selectedParentId: string | null = null // ID of the selected target folder
	let selectedParentName: string = 'Root' // Name for display

	async function fetchTree() {
		isTreeLoading = true
		treeError = null
		console.log('Fetching document tree...')
		try {
			// Use relative path, assuming same origin or proxy
			const response = await fetch('/api/documents/tree')
			if (!response.ok) {
				let errorDetail = 'Failed to fetch document tree.'
				try {
					const errorData = await response.json()
					errorDetail = errorData.detail || errorDetail
				} catch (jsonError) {
					/* Ignore */
				}
				throw new Error(`HTTP error ${response.status}: ${errorDetail}`)
			}
			documentTree = await response.json()
			console.log('Document tree fetched:', documentTree)
		} catch (error: any) {
			console.error('Error fetching document tree:', error)
			treeError = error.message || 'Could not fetch document tree.'
			documentTree = [] // Clear tree on error
		} finally {
			isTreeLoading = false
		}
	}

	// Helper function to build the tree structure
	function buildTree(items: any[]): any[] {
		const tree: any[] = []
		const map: { [key: string]: any } = {}

		// First pass: create map and add children array
		items.forEach(item => {
			map[item._id] = { ...item, children: [] }
		})

		// Second pass: build the tree structure
		items.forEach(item => {
			const node = map[item._id]
			if (item.parent_id && map[item.parent_id]) {
				// If it has a parent in the map, add it to the parent's children
				map[item.parent_id].children.push(node)
			} else {
				// Otherwise, it's a root node
				tree.push(node)
			}
		})

		// Optional: Sort children alphabetically by filename?
		// Object.values(map).forEach(node => {
		//   node.children.sort((a: any, b: any) => a.filename.localeCompare(b.filename));
		// });
		// tree.sort((a: any, b: any) => a.filename.localeCompare(b.filename));

		return tree
	}

	$: renderedTree = buildTree(documentTree) // Reactive statement to rebuild tree when data changes

	onMount(() => {
		fetchTree() // Fetch tree when component mounts
	})

	// Handler for node selection change from TreeNode component
	async function handleNodeSelect(
		event: CustomEvent<{ id: string; selected: boolean }>
	) {
		const { id, selected } = event.detail
		console.log(`Node ${id} selected: ${selected}`)
		// TODO: Add visual feedback for loading state during API call

		try {
			const response = await fetch('/api/documents/select', {
				method: 'PUT',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify({ document_id: id, selected: selected }),
			})

			if (!response.ok) {
				let errorDetail = 'Failed to update selection status.'
				try {
					const errorData = await response.json()
					errorDetail = errorData.detail || errorDetail
				} catch (jsonError) {
					/* Ignore */
				}
				throw new Error(`HTTP error ${response.status}: ${errorDetail}`)
			}

			const result = await response.json()
			console.log('Selection update result:', result.message)

			// Option 1: Update local state directly (more complex to ensure consistency)
			// function updateNodeState(nodes: any[], targetId: string, selectedState: boolean) {
			// 	for (const node of nodes) {
			// 		if (node._id === targetId) {
			// 			node.selected_for_rag = selectedState;
			// 			return true; // Found and updated
			// 		}
			// 		if (node.children && node.children.length > 0) {
			// 			if (updateNodeState(node.children, targetId, selectedState)) {
			// 				return true; // Found in children
			// 			}
			// 		}
			// 	}
			// 	return false; // Not found in this branch
			// }
			// if (updateNodeState(documentTree, id, selected)) {
			// 	documentTree = [...documentTree]; // Trigger reactivity
			// }

			// Option 2: Refetch the entire tree (simpler, ensures consistency)
			await fetchTree()
		} catch (error: any) {
			console.error('Error updating selection:', error)
			// Display error to user (e.g., using uploadMessage or a dedicated tree message state)
			uploadMessage = {
				type: 'error',
				text: `Failed to update selection: ${error.message}`,
			}
		}
	}

	// Handler for folder selection from TreeNode component
	function handleFolderSelect(
		event: CustomEvent<{ id: string; name: string }>
	) {
		selectedParentId = event.detail.id
		selectedParentName = event.detail.name
		console.log(
			`Selected target folder: ${selectedParentName} (ID: ${selectedParentId})`
		)
	}

	async function handleUpload() {
		if (!files || files.length === 0) {
			// Use simple message state instead of toast
			uploadMessage = {
				// Fix: Use uploadMessage
				type: 'warning',
				text: 'Please select one or more files to upload.',
			}
			return
		}

		isUploading = true // Fix: Use isUploading
		uploadProgress = 0 // Reset progress
		uploadMessage = null // Fix: Use uploadMessage & Clear previous messages

		const formData = new FormData()
		for (let i = 0; i < files.length; i++) {
			formData.append('files', files[i])
		}
		// Add selected parent_id if one is chosen
		if (selectedParentId) {
			formData.append('parent_id', selectedParentId)
		}

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
			uploadMessage = {
				// Fix: Use uploadMessage
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
			uploadMessage = {
				// Fix: Use uploadMessage
				type: 'error',
				text: `Upload failed: ${error.message || 'Unknown error'}`,
			}
		} finally {
			isUploading = false // Fix: Use isUploading
		}
	}
</script>

<div class="container mx-auto p-10 space-y-4">
	<h1 class="text-2xl font-bold mb-4">Upload Documents</h1>

	<!-- Simple Message Display -->
	{#if uploadMessage}
		<div
			class="p-4 rounded-md text-sm"
			class:bg-green-100={uploadMessage.type === 'success'}
			class:text-green-700={uploadMessage.type === 'success'}
			class:dark:bg-green-900={uploadMessage.type === 'success'}
			class:dark:text-green-200={uploadMessage.type === 'success'}
			class:bg-red-100={uploadMessage.type === 'error'}
			class:text-red-700={uploadMessage.type === 'error'}
			class:dark:bg-red-900={uploadMessage.type === 'error'}
			class:dark:text-red-200={uploadMessage.type === 'error'}
			class:bg-yellow-100={uploadMessage.type === 'warning'}
			class:text-yellow-700={uploadMessage.type === 'warning'}
			class:dark:bg-yellow-900={uploadMessage.type === 'warning'}
			class:dark:text-yellow-200={uploadMessage.type === 'warning'}
		>
			{uploadMessage.text}
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
					disabled={isUploading}
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
				<span>Target Folder</span>
				<input
					type="text"
					class="block w-full p-2.5 mt-1 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-gray-700 dark:text-gray-300"
					value={selectedParentName}
					readonly
				/>
				{#if selectedParentId}
					<button
						type="button"
						on:click={() => {
							selectedParentId = null
							selectedParentName = 'Root'
						}}
						class="text-xs text-blue-600 hover:underline mt-1"
						>Clear (Upload to Root)</button
					>
				{/if}
				{#if selectedParentId}
					<button
						type="button"
						on:click={() => {
							selectedParentId = null
							selectedParentName = 'Root'
						}}
						class="text-xs text-blue-600 hover:underline mt-1"
						>Clear (Upload to Root)</button
					>
				{/if}
			</label>
		</div>

		<button
			type="button"
			class="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
			on:click={handleUpload}
			disabled={isUploading || !files || files.length === 0}
		>
			{#if isUploading}
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

	<!-- Tree View Display Area -->
	<div
		class="p-4 border border-gray-300 dark:border-gray-700 rounded-lg bg-white dark:bg-gray-800 shadow-md"
	>
		<h2 class="text-xl font-semibold mb-2 text-gray-900 dark:text-white">
			Document Tree
		</h2>
		{#if isTreeLoading}
			<p class="text-sm text-gray-600 dark:text-gray-400">Loading tree...</p>
		{:else if treeError}
			<div
				class="p-4 rounded-md text-sm bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-200"
			>
				Error loading tree: {treeError}
			</div>
		{:else if documentTree.length === 0}
			<p class="text-sm text-gray-600 dark:text-gray-400">
				No documents found.
			</p>
		{:else}
			<!-- Render the tree using the TreeNode component -->
			<div class="document-tree">
				{#each renderedTree as rootNode (rootNode._id)}
					<TreeNode
						node={rootNode}
						children={rootNode.children || []}
						level={0}
						on:select={handleNodeSelect}
						on:selectFolder={handleFolderSelect}
					/>
				{/each}
			</div>
		{/if}
	</div>
</div>
