# Bundled furnace diagrams — licences and attribution

These files are displayed in the Blend Mix Optimizer result panel
(`src/ui/bmo/components.py::_render_furnace_view`).

They are **bundled rather than hotlinked** so the page renders identically on a
plant machine with no internet egress, and so a remote host changing a URL can
never put a broken image in front of an operator.

Both licences require attribution wherever the image is shown. That attribution
is rendered in the caption beneath each image in the app — **do not remove it**,
and if you move the images to another page, carry the captions with them.

---

## blast_furnace_zone_reactions.jpg

| | |
|---|---|
| Title | Blast Furnace Reactions |
| Author | OpenStax |
| Licence | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| Source | https://commons.wikimedia.org/wiki/File:Blast_Furnace_Reactions.jpg |
| Retrieved | 2026-09-03 |
| Used | unmodified |

Shows the reduction chemistry by furnace zone with heights and temperatures.
Chosen because its reactions are the ones this application actually models:
Boudouard regeneration (C + CO₂ → 2CO), flux calcination (CaCO₃ → CaO + CO₂,
the flux CO₂ term in the coke correction) and slag formation
(CaO + SiO₂ → CaSiO₃).

**Share-alike note:** CC BY-SA applies to derivative works. Displaying the image
unmodified alongside our own content does not make the application a derivative,
so no copyleft obligation attaches to the surrounding code. If someone edits the
image itself, the edited version must be released under CC BY-SA 4.0.

---

## blast_furnace_cross_section.png

| | |
|---|---|
| Title | Blast furnace NT |
| Author | Tosaka (2008), revised by Vussiewussie (2009) |
| Licence | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| Source | https://commons.wikimedia.org/wiki/File:Blast_furnace_NT.PNG |
| Retrieved | 2026-09-03 |
| Used | unmodified |

Numbered plant layout — skip hoist, bell, stack, tuyeres, stoves, dust catcher,
taphole. The numbers are not labelled in the image itself; the caption in the
app names the ones that matter for this page.

---

## If you add another image here

Record it in this file with author, licence, source URL and retrieval date
**before** it goes on a page, and put the attribution in the on-screen caption.
Prefer public domain, CC0, CC BY or CC BY-SA from Wikimedia Commons. Do not add
images found through a general image search without confirming the licence on
the originating page — search-engine thumbnails carry no licence grant.
