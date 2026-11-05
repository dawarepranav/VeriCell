# Runway prompts (b-roll only)

Runway makes the mood shots. It must **never** generate anything that looks like a VeriCell result:
no cell outlines, no numbers, no UI, no "microscope images" presented as data. Those come only from
the screen recording. Say "b-roll generated with Runway" in the video description.

**Settings for every shot:** 16:9, 1080p (or the highest your plan allows), 5 s unless noted, no
on-screen text. Generate 2–3 takes per shot and keep the calmest one.

**House style (append to every prompt):**
`cinematic, shallow depth of field, cool clinical light with a single warm amber accent, deep navy
shadows, slow and steady camera, photorealistic, no text, no logos, no people's faces`

---

## RW-1 · Opening (0:00–0:04), required

> Extreme close-up of an inverted microscope's objective lens beneath a clear plastic multiwell
> plate, each well holding pale pink culture medium. The camera pushes in slowly toward one well
> as the light brightens to a soft white brightfield glow. Quiet, precise, laboratory at night.

- Duration: 5 s. Cut at about 4 s, hard cut to the real Colon image from the screen recording.
- Avoid: visible cells or blobs inside the well (the real image follows), lab coats, screens.

## RW-2 · Transition into the image (optional, 0:03–0:04)

> Looking down a microscope eyepiece, the circular field of view fills with soft out-of-focus
> brightfield haze that slowly sharpens to plain white light. No objects in the field.

- Duration: 5 s; use the last 1 s as a white flash into the real image.

## RW-3 · Behind the provenance card (optional, 0:07–0:15)

> Out-of-focus laboratory at night: bokeh of small instrument lights in navy and amber, a
> microscope silhouette at the edge of frame, very slow lateral drift.

- Duration: 10 s. Keep it dark; the card text sits on top at 40–60% card opacity.

## RW-4 · Closing (2:45–2:50), required

> A lab bench at dusk: a gloved hand slides a multiwell plate onto a microscope stage, a laptop
> glows softly in the background with an unreadable blurred screen, warm amber window light
> fading to navy. Camera slowly pulls back.

- Duration: 5 s. Cross-fade into `cards/07-close.png`.
- Avoid: readable screen content, logos on the laptop, faces.

---

## If you want motion on the real image (instead of Runway)

Don't send the microscope image to image-to-video: generators invent or move cells, which would
misrepresent data. Animate it in the editor instead: a slow 105% → 115% scale ("Ken Burns") on
`stills/colon-raw.png` over 4 seconds is enough.
