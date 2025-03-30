// frontend/tailwind.config.ts
import { join } from 'path'
import type { Config } from 'tailwindcss'
import forms from '@tailwindcss/forms'
import typography from '@tailwindcss/typography'
// Skeleton plugin removed

const config = {
	// Opt for dark mode to be handled via the class method
	darkMode: 'class',
	content: [
		'./src/**/*.{html,js,svelte,ts}',
		// Append the path to the Skeleton package to load Skeleton's CSS classes
		join(
			require.resolve('@skeletonlabs/skeleton'),
			'../**/*.{html,js,svelte,ts}'
		),
	],
	theme: {
		extend: {},
	},
	plugins: [
		forms,
		typography,
		// Skeleton plugin removed
	],
} satisfies Config

export default config
