/*
 * ZPix Gradio app custom JavaScript.
 */

// Delegate events to work with Gradio rendering.
document.addEventListener("click", (event) => {
    if (event.target.closest("#swap-lora-btn")) {
        return swapLora()
    }
})
document.addEventListener("mouseover", (event) => {
    if (event.target.closest("#prompt textarea")) {
        const prompt = document.getElementById("prompt")
        if (prompt?.title) hidePromptTooltip(prompt)
    }
})
document.addEventListener("mouseleave", (event) => {
    if (event.target.closest?.("#prompt textarea")) {
        const prompt = document.getElementById("prompt")
        if (prompt?.dataset.title) restorePromptTooltip(prompt)
    }
}, true) // Required because mouseleave doesn't bubble.

/**
 * Hide the prompt tooltip on its textarea mouseover
 * so tooltip doesn't cause inconvenience to the user.
 * @param {HTMLElement} prompt
 */
function hidePromptTooltip(prompt) {
    prompt.dataset.title = prompt.title
    prompt.title = ""
}

/**
 * Restore the prompt tooltip on its textarea mouseleave.
 * @param {HTMLElement} prompt
 */
function restorePromptTooltip(prompt) {
    prompt.title = prompt.dataset.title
    delete prompt.dataset.title
}

/**
 * Swap LoRA.
 */
async function swapLora() {
    /**
     * Absolute path to LoRA file selected by user.
     * @type {string}
     */
    const path = await window.openNativeFileDialog()
    // This avoids an unnecessary local upload. See webview.cpp

    if (!path) return // When the user cancels.

    /** @type {HTMLTextAreaElement} */
    const portal = document.querySelector("#lora-path textarea")

    // We place the LoRA path in a "portal" hidden textarea.
    // This "portal" transfers data and control to the backend.
    // Appending a timestamp forces the change event to fire
    // when the user loads-unloads-reloads the same LoRA file.
    portal.value = `${path}|${Math.floor(Date.now() / 1000)}`
    portal.dispatchEvent(new Event("input"))
}