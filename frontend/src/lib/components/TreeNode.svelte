<!-- frontend/src/lib/components/TreeNode.svelte -->
<script lang="ts">
	import { createEventDispatcher } from 'svelte'

	export let node: any // The current node data (document/folder)
	export let children: any[] = [] // Children of the current node
	export let level = 0 // Indentation level

	const dispatch = createEventDispatcher<{
		select: { id: string; selected: boolean }
		selectFolder: { id: string; name: string } // Add selectFolder event type
	}>() // Typed dispatcher

	// Handle checkbox change
	function handleSelectChange(event: Event) {
		const target = event.target as HTMLInputElement
		dispatch('select', { id: node._id, selected: target.checked })
	}

	// Handle folder click for target selection
	function selectFolder() {
		if (node.is_folder) {
			console.log('Folder selected as target:', node.filename, node._id)
			dispatch('selectFolder', { id: node._id, name: node.filename })
		}
	}

	$: isFolder = node.is_folder === true
	$: indentStyle = `padding-left: ${level * 1.5}rem;` // Indentation based on level
</script>

<div
	class="tree-node my-1"
	style={indentStyle}
>
	<div class="flex items-center space-x-2">
		<!-- Checkbox for RAG selection -->
		<input
			type="checkbox"
			bind:checked={node.selected_for_rag}
			on:change={handleSelectChange}
			class="checkbox checkbox-sm mr-2"
		/>

		<span>
			{#if isFolder}
				<!-- Folder Icon (simple) -->
				<svg
					xmlns="http://www.w3.org/2000/svg"
					class="h-5 w-5 inline-block mr-1 text-yellow-500"
					viewBox="0 0 20 20"
					fill="currentColor"
				>
					<path
						d="M2 6a2 2 0 012-2h5l2 2h5a2 2 0 012 2v6a2 2 0 01-2 2H4a2 2 0 01-2-2V6z"
					/>
				</svg>
			{:else}
				<!-- File Icon (simple) -->
				<svg
					xmlns="http://www.w3.org/2000/svg"
					class="h-5 w-5 inline-block mr-1 text-gray-500"
					viewBox="0 0 20 20"
					fill="currentColor"
				>
					<path
						fill-rule="evenodd"
						d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z"
						clip-rule="evenodd"
					/>
				</svg>
			{/if}
			<!-- Make folder name clickable -->
			{#if isFolder}
				<button
					on:click={selectFolder}
					class="text-left hover:underline focus:outline-none font-semibold"
				>
					{node.filename}
				</button>
			{:else}
				<span>{node.filename}</span>
			{/if}
			{#if !isFolder}
				<span class="text-xs text-gray-400 ml-2">({node.mimetype})</span>
			{/if}
		</span>

		<!-- TODO: Add button to select folder as target -->
		<!-- {#if isFolder}
      <button class="btn btn-xs btn-outline" on:click={selectFolder}>Select Target</button>
    {/if} -->
	</div>

	<!-- Recursive Rendering for Children -->
	{#if children.length > 0}
		<div class="children mt-1">
			{#each children as childNode (childNode._id)}
				<!-- Pass the node and its children down recursively -->
				<svelte:self
					node={childNode}
					children={childNode.children || []}
					level={level + 1}
				/>
			{/each}
		</div>
	{/if}
</div>
