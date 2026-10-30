<!--
The root never renders: its load (`+page.ts`) redirects to the landing page, Furnace status (SV-7.2, #325). The file
keeps `/` a page route, so the redirect runs after the layout's session check.
-->
<p class="sr-only">Opening Furnace status.</p>
