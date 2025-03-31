<script lang="ts">
	// Basic script setup for the chat page
	import { onMount } from 'svelte'

	let messages: { role: 'user' | 'assistant'; content: string }[] = []
	let userInput = ''
	let isLoading = false

	async function sendMessage() {
		if (!userInput.trim() || isLoading) return

		const newUserMessage = { role: 'user' as const, content: userInput }
		messages = [...messages, newUserMessage]
		const currentInput = userInput
		userInput = ''
		isLoading = true

		try {
			console.log('Sending message to backend:', currentInput)
			const response = await fetch('/api/chat', {
				// Relative URL, assumes frontend/backend on same origin or proxied
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
				},
				body: JSON.stringify({ message: currentInput }),
			})

			if (!response.ok) {
				let errorDetail = 'Failed to get response from backend.'
				try {
					const errorData = await response.json()
					errorDetail = errorData.detail || errorDetail
				} catch (jsonError) {
					// Ignore if response is not JSON
				}
				throw new Error(`HTTP error ${response.status}: ${errorDetail}`)
			}

			const data = await response.json()
			const assistantResponse = {
				role: 'assistant' as const,
				content: data.response,
			}
			// TODO: Optionally display sources if data.sources is present

			messages = [...messages, assistantResponse]
		} catch (error: any) {
			console.error('Error sending message:', error)
			const errorResponse = {
				role: 'assistant' as const,
				content: `Error: ${error.message || 'Could not communicate with backend.'}`,
			}
			messages = [...messages, errorResponse]
		} finally {
			isLoading = false
		}
	}

	onMount(() => {
		console.log('Chat page mounted')
	})
</script>

<div class="flex flex-col h-full p-4">
	<h1 class="text-2xl font-bold mb-4">Chat with your Documents</h1>

	<div class="flex-grow overflow-y-auto mb-4 border rounded p-2 space-y-2">
		{#each messages as message (message)}
			<div class:text-right={message.role === 'user'}>
				<span
					class="inline-block p-2 rounded-lg"
					class:bg-blue-100={message.role === 'user'}
					class:bg-gray-200={message.role === 'assistant'}
				>
					{message.content}
				</span>
			</div>
		{/each}
		{#if isLoading}
			<div class="text-gray-500">Assistant is thinking...</div>
		{/if}
	</div>

	<div class="flex">
		<input
			type="text"
			bind:value={userInput}
			placeholder="Ask a question..."
			class="flex-grow input input-bordered mr-2"
			on:keydown={e => e.key === 'Enter' && sendMessage()}
			disabled={isLoading}
		/>
		<button
			class="btn btn-primary"
			on:click={sendMessage}
			disabled={isLoading}
		>
			Send
		</button>
	</div>
</div>

<style>
	/* Add any specific styles if needed, Tailwind is used via app.css */
</style>
