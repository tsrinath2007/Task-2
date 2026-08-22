let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordStartTime;
let timerInterval;
let latestAudioBlob = null; // Stably store recorded blob to prevent closure bugs

// Translation State
let originalAnswerText = "";
let translatedAnswerText = "";
let isCurrentlyTranslated = false;

// DOM Elements
const translateBtn = document.getElementById("translate-btn");
const recordBtn = document.getElementById("record-btn");
const recordIcon = document.getElementById("record-icon");
const recordStatus = document.getElementById("record-status");
const recordingTime = document.getElementById("recording-time");
const audioPlayback = document.getElementById("audio-playback");
const queryTextInput = document.getElementById("query-text-input");
const textSubmitBtn = document.getElementById("text-submit-btn");
const strategySelect = document.getElementById("strategy");
const thresholdInput = document.getElementById("threshold");
const thresholdVal = document.getElementById("threshold-val");
const outResponse = document.getElementById("out-response");
const outCoreLatency = document.getElementById("out-core-latency");
const outTotalLatency = document.getElementById("out-total-latency");
const guardrailBadges = document.getElementById("guardrail-badges");
const chunksContainer = document.getElementById("chunks-container");
const p50Val = document.getElementById("p50-val");
const p70Val = document.getElementById("p70-val");
const p100Val = document.getElementById("p100-val");
const resetBtn = document.getElementById("reset-btn");
const outputCard = document.getElementById("output-card");
const waveform = document.getElementById("waveform");

// Initialize threshold label
if (thresholdInput && thresholdVal) {
    thresholdInput.addEventListener("input", (e) => {
        thresholdVal.textContent = e.target.value;
    });
}

let speechRecognizer = null;
let liveSpeechTranscript = "";

// Initialize WebSpeechAPI if supported by browser
function setupSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        try {
            speechRecognizer = new SpeechRecognition();
            speechRecognizer.continuous = true;
            speechRecognizer.interimResults = true;
            
            const languageSelect = document.getElementById("language-select");
            speechRecognizer.lang = languageSelect ? languageSelect.value : "en-US";

            if (languageSelect) {
                languageSelect.addEventListener("change", () => {
                    if (speechRecognizer) {
                        speechRecognizer.lang = languageSelect.value;
                    }
                });
            }

            speechRecognizer.onresult = (event) => {
                let current = "";
                for (let i = 0; i < event.results.length; i++) {
                    current += event.results[i][0].transcript + " ";
                }
                if (current.trim()) {
                    liveSpeechTranscript = current.trim();
                    if (queryTextInput) queryTextInput.value = liveSpeechTranscript;
                }
            };

            speechRecognizer.onerror = (err) => {
                console.warn("SpeechRecognition error:", err);
            };
        } catch (e) {
            console.warn("SpeechRecognition init failed:", e);
        }
    }
}

// Setup audio recorder
async function setupRecorder() {
    setupSpeechRecognition();
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        let mimeType = "audio/webm";
        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
            mimeType = "audio/webm;codecs=opus";
        } else if (MediaRecorder.isTypeSupported("audio/webm")) {
            mimeType = "audio/webm";
        } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
            mimeType = "audio/mp4";
        }

        mediaRecorder = new MediaRecorder(stream, { mimeType });
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data && event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            // Securely create and store the latest audio blob
            const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
            audioChunks = [];
            latestAudioBlob = audioBlob;
            
            // Set audio player playback source
            const audioUrl = URL.createObjectURL(latestAudioBlob);
            audioPlayback.src = audioUrl;
            audioPlayback.classList.remove("hidden");

            // Submit text directly if liveSpeechTranscript or queryTextInput is present
            const currentInputValue = queryTextInput ? queryTextInput.value.trim() : "";
            const queryTextToSubmit = liveSpeechTranscript || currentInputValue;

            if (queryTextToSubmit) {
                liveSpeechTranscript = "";
                await submitPipelineQuery(queryTextToSubmit, null);
            } else {
                await submitPipelineQuery(null, latestAudioBlob);
            }
        };
    } catch (err) {
        console.error("Error accessing microphone:", err);
        recordStatus.textContent = "Mic access blocked";
        recordBtn.disabled = true;
        recordBtn.classList.remove("bg-[#FFC93C]");
        recordBtn.classList.add("bg-gray-700", "text-gray-400", "cursor-not-allowed");
    }
}

// Start recording (updates styles to active red-pink pill, starts visual wave animation)
function startRecording() {
    isRecording = true;
    audioChunks = [];
    latestAudioBlob = null;
    liveSpeechTranscript = "";
    
    if (speechRecognizer) {
        const languageSelect = document.getElementById("language-select");
        if (languageSelect) speechRecognizer.lang = languageSelect.value;
        try { speechRecognizer.start(); } catch(e) {}
    }
    
    mediaRecorder.start(100);
    
    // UI state: Turn to warning pink-red, start waveform pulse
    recordBtn.classList.remove("bg-[#FFC93C]", "text-[#0F3D2E]", "hover:bg-[#ffe180]");
    recordBtn.classList.add("bg-[#FF2E7E]", "text-white", "hover:bg-[#ff5294]", "animate-pulse");
    recordIcon.className = "fa-solid fa-square text-lg";
    recordStatus.textContent = "Stop Listening";
    recordingTime.classList.remove("hidden");
    if (waveform) waveform.classList.remove("hidden");
    audioPlayback.classList.add("hidden");
    
    // Timer
    recordStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
        const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const secs = String(elapsed % 60).padStart(2, '0');
        recordingTime.textContent = `${mins}:${secs}`;
    }, 1000);
}

// Stop recording (reverts back to gold pill, hides waveform)
function stopRecording() {
    isRecording = false;
    if (speechRecognizer) {
        try { speechRecognizer.stop(); } catch(e) {}
    }
    mediaRecorder.stop();
    
    // Revert UI State back to Gold
    recordBtn.classList.remove("bg-[#FF2E7E]", "text-white", "hover:bg-[#ff5294]", "animate-pulse");
    recordBtn.classList.add("bg-[#FFC93C]", "text-[#0F3D2E]", "hover:bg-[#ffe180]");
    recordIcon.className = "fa-solid fa-microphone text-lg";
    recordStatus.textContent = "Transcribing...";
    recordingTime.classList.add("hidden");
    if (waveform) waveform.classList.add("hidden");
    clearInterval(timerInterval);
}

// Mic button click binding
if (recordBtn) {
    recordBtn.addEventListener("click", () => {
        if (!mediaRecorder) {
            setupRecorder().then(() => {
                if (mediaRecorder) startRecording();
            });
            return;
        }
        
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    });
}

// Submit text query binding
if (textSubmitBtn) {
    textSubmitBtn.addEventListener("click", () => {
        const text = queryTextInput.value.trim();
        if (text) {
            submitPipelineQuery(text, null);
        }
    });
}

if (queryTextInput) {
    queryTextInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            const text = queryTextInput.value.trim();
            if (text) submitPipelineQuery(text, null);
        }
    });
}

// Core API Query Trigger
async function submitPipelineQuery(text, audioBlob) {
    // Show spinner in response card
    outResponse.innerHTML = `<div class="flex items-center gap-2 text-[#FFC93C] font-semibold"><i class="fa-solid fa-spinner animate-spin"></i> Analyzing pipeline path...</div>`;
    guardrailBadges.innerHTML = "";
    
    const formData = new FormData();
    const languageSelect = document.getElementById("language-select");
    formData.append("strategy", strategySelect ? strategySelect.value : "sentence-aware");
    formData.append("off_topic_threshold", thresholdInput ? thresholdInput.value : 0.35);
    formData.append("language", languageSelect ? languageSelect.value : "en-US");
    
    if (text) {
        formData.append("query_text", text);
    } else if (audioBlob) {
        formData.append("file", audioBlob, "query.webm");
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25000);

    try {
        const res = await fetch("/api/query", {
            method: "POST",
            body: formData,
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        
        if (!res.ok) {
            const errDetail = await res.json().catch(() => ({ detail: "Server error occurred" }));
            throw new Error(errDetail.detail || "Server error occurred");
        }
        
        const data = await res.json();
        renderPipelineResponse(data, audioBlob);
        await updateAnalytics();
    } catch (err) {
        clearTimeout(timeoutId);
        if (err.name === "AbortError") {
            outResponse.innerHTML = `<span class="text-[#FF2E7E] font-semibold"><i class="fa-solid fa-clock"></i> Request timed out (25s limit reached). Please retry your query.</span>`;
        } else {
            outResponse.innerHTML = `<span class="text-[#FF2E7E] font-semibold"><i class="fa-solid fa-circle-exclamation"></i> Error: ${err.message}</span>`;
        }
        if (audioBlob && recordStatus) recordStatus.textContent = "Query timed out";
    }
}

// Render dynamic responses and badges
function renderPipelineResponse(data, audioBlob = null) {
    // Status border glows
    outputCard.className = "lg:col-span-12 glass-card p-8 transition-all duration-300";
    if (data.status === "refused") {
        outputCard.classList.add("border-[#FF2E7E]/40", "glow-pink");
        outResponse.className = "text-[#FF2E7E] text-lg font-medium leading-relaxed";
    } else if (data.status === "success") {
        outputCard.classList.add("border-[#2ED9A0]/40", "glow-mint");
        outResponse.className = "text-[#F3F1E7] text-lg font-medium leading-relaxed";
    }
    outResponse.textContent = data.response_text;

    // Display transcribed voice input in the text input box and response badge
    const voiceContainer = document.getElementById("voice-transcript-container");
    const voiceText = document.getElementById("voice-transcript-text");
    
    if (data.query_text && data.query_text !== "No speech detected in audio.") {
        if (queryTextInput) queryTextInput.value = data.query_text;
        if (voiceContainer && voiceText) {
            voiceText.textContent = `"${data.query_text}"`;
            voiceContainer.classList.remove("hidden");
        }
    } else if (voiceContainer) {
        voiceContainer.classList.add("hidden");
    }

    // Reset translation state for new query response
    originalAnswerText = data.response_text;
    translatedAnswerText = "";
    isCurrentlyTranslated = false;
    
    if (translateBtn) {
        if (data.status === "success") {
            const isHindi = /[\u0900-\u097F]/.test(data.response_text);
            translateBtn.textContent = isHindi ? "Translate to English" : "Translate to Hindi";
            translateBtn.classList.remove("hidden");
        } else {
            translateBtn.classList.add("hidden");
        }
    }

    // Latency details
    const timings = data.latency_breakdown;
    const coreLatency = timings.embed_ms + timings.retrieve_ms + timings.llm_generate_ms + timings.guardrails_ms;

    // STT Provider Badge update
    const sttProviderBadge = document.getElementById("stt-provider-badge");
    if (sttProviderBadge && timings.stt_provider) {
        sttProviderBadge.textContent = `STT: ${timings.stt_provider}`;
    }

    // Timing status badge ONLINE
    const timingStatusBadge = document.getElementById("timing-status-badge");
    if (timingStatusBadge) {
        timingStatusBadge.textContent = `ONLINE (${coreLatency.toFixed(1)} MS)`;
        timingStatusBadge.classList.remove("hidden");
    }

    // Pipeline Sub-Metrics text updates
    document.getElementById("metric-stt").textContent = `${timings.stt_ms.toFixed(1)} ms`;
    document.getElementById("metric-retrieval").textContent = `${timings.retrieve_ms.toFixed(1)} ms`;
    document.getElementById("metric-generation").textContent = `${timings.llm_generate_ms.toFixed(1)} ms`;
    document.getElementById("metric-total").textContent = `${timings.total_ms.toFixed(1)} ms`;

    // Dynamic Guardrail Status Pills (top right of response card)
    const guards = data.guardrail_results;
    let badgesHtml = "";
    
    // Prominent Groundedness Status Pill
    if (data.status === "success") {
        badgesHtml += `<span class="px-3.5 py-1 bg-[#2ED9A0]/20 text-[#2ED9A0] border border-[#2ED9A0]/45 rounded-full text-xs font-bold font-mono tracking-wider flex items-center gap-1.5 glow-mint">
            <span class="h-2 w-2 rounded-full bg-[#2ED9A0]"></span> GROUNDED
        </span>`;
    } else if (guards.off_topic) {
        badgesHtml += `<span class="px-3.5 py-1 bg-[#FF2E7E]/20 text-[#FF2E7E] border border-[#FF2E7E]/45 rounded-full text-xs font-bold font-mono tracking-wider flex items-center gap-1.5 glow-pink">
            <span class="h-2 w-2 rounded-full bg-[#FF2E7E]"></span> OFF-TOPIC
        </span>`;
    } else if (guards.grounded === false) {
        badgesHtml += `<span class="px-3.5 py-1 bg-[#FFC93C]/20 text-[#FFC93C] border border-[#FFC93C]/45 rounded-full text-xs font-bold font-mono tracking-wider flex items-center gap-1.5 glow-gold">
            <span class="h-2 w-2 rounded-full bg-[#FFC93C]"></span> INSUFFICIENT EVIDENCE
        </span>`;
    }
    
    // Safety flag tag
    if (!guards.safe) {
        badgesHtml += `<span class="px-2.5 py-1 bg-[#FF2E7E]/20 text-[#FF2E7E] border border-[#FF2E7E]/40 rounded-full text-[10px] font-bold font-mono uppercase tracking-wider">UNSAFE</span>`;
    }
    guardrailBadges.innerHTML = badgesHtml;

    // Grounded Answer Sub-Badges (Confidence and Evidence)
    const confBadge = document.getElementById("confidence-badge");
    const evidBadge = document.getElementById("evidence-badge");
    const evidContainer = document.getElementById("evidence-container");
    const evidCapsule = document.getElementById("evidence-capsule");

    if (data.status === "success" && data.retrieved_chunks && data.retrieved_chunks.length > 0) {
        confBadge.classList.remove("hidden");
        evidBadge.classList.remove("hidden");
        
        // Show EVIDENCE row and top chunk reference hash
        const firstChunk = data.retrieved_chunks[0];
        const cleanId = String(firstChunk.query_id || "81121b80").substring(0, 8);
        const passageIdx = firstChunk.passage_index !== undefined ? firstChunk.passage_index : 0;
        
        evidCapsule.textContent = `[${passageIdx + 1}] ${cleanId}...`;
        evidContainer.classList.remove("hidden");
    } else {
        confBadge.classList.add("hidden");
        evidBadge.classList.add("hidden");
        evidContainer.classList.add("hidden");
    }

    // Chunks Header Badges (Similarity / Count / Time)
    const chunksCount = document.getElementById("chunks-count");
    const maxSimBadge = document.getElementById("max-similarity-badge");
    const retTimeBadge = document.getElementById("retrieval-time-badge");
    
    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
        chunksCount.textContent = data.retrieved_chunks.length;
        const maxScore = data.retrieved_chunks[0].score;
        
        maxSimBadge.textContent = `Max Similarity: ${maxScore.toFixed(3)}`;
        maxSimBadge.classList.remove("hidden");
        
        retTimeBadge.textContent = `Retrieval: ${timings.retrieve_ms.toFixed(1)} ms`;
        retTimeBadge.classList.remove("hidden");
    } else {
        chunksCount.textContent = "0";
        maxSimBadge.classList.add("hidden");
        retTimeBadge.classList.add("hidden");
    }

    // Display retrieved passages with colored strategy tags matching reference
    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
        chunksContainer.innerHTML = data.retrieved_chunks.map((chunk, idx) => {
            // Determine strategy badge style
            let strategyTag = "Fixed Overlap (512/128)";
            let strategyColor = "bg-[#3B82F6]";
            
            if (chunk.strategy === "sentence-aware") {
                strategyTag = "Semantic Split (Cos > 0.35)";
                strategyColor = "bg-[#0D9488]";
            } else if (chunk.strategy === "structure-aware") {
                strategyTag = "Metadata-Aware";
                strategyColor = "bg-[#D97706]";
            }
            
            return `
                <div class="p-5 bg-[#0C3426]/40 border ${chunk.is_selected ? 'border-[#FFC93C]/40 bg-[#FFC93C]/10 shadow-[#FFC93C]/10' : 'border-white/10'} rounded-2xl backdrop-blur-md transition hover:border-white/20 duration-200">
                    <div class="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-2 mb-3">
                        <div class="flex flex-wrap items-center gap-2">
                            <span class="px-2.5 py-0.5 bg-white/10 border border-white/15 rounded-lg text-[10px] font-mono font-bold text-white">#${idx + 1}</span>
                            <span class="px-2.5 py-0.5 rounded-lg text-[10px] font-mono font-bold text-white ${strategyColor} flex items-center gap-1.5">
                                <span class="h-1.5 w-1.5 rounded-full bg-white"></span> ${strategyTag.toUpperCase()}
                            </span>
                            ${chunk.is_selected ? '<span class="px-2.5 py-0.5 border border-[#3B82F6] text-[#60A5FA] bg-[#3B82F6]/10 rounded-lg text-[9px] font-mono font-bold uppercase tracking-wider">CITED BY LLM</span>' : ''}
                        </div>
                        <div class="text-left sm:text-right font-mono text-xs">
                            <span class="text-[#9FB8AC] mr-3">Score: <span class="text-[#FFC93C] font-bold">${chunk.score.toFixed(3)}</span></span>
                            <span class="text-[#9FB8AC]">Parent: <span class="text-[#F3F1E7] font-semibold">${String(chunk.query_id || "1102432").substring(0, 8)}:${chunk.passage_index}</span></span>
                        </div>
                    </div>
                    <p class="text-sm text-[#F3F1E7] leading-relaxed font-body mb-3">${chunk.text}</p>
                    <div class="text-left">
                        <span class="text-[10px] font-mono font-bold text-[#FFC93C] hover:text-[#ffe180] tracking-wider uppercase cursor-pointer transition select-none">Click to view full text</span>
                    </div>
                </div>
            `;
        }).join("");
    } else {
        chunksContainer.innerHTML = `<p class="text-[#9FB8AC] italic text-sm text-center py-8">No passages retrieved.</p>`;
    }

    // Update RAG target latency bar (represented relative to 300ms limit)
    const latencyBarFill = document.getElementById("latency-bar-fill");
    if (latencyBarFill) {
        let percentage = Math.min((coreLatency / 300) * 100, 100);
        latencyBarFill.style.width = percentage + "%";
        
        // Remove coloring classes
        latencyBarFill.classList.remove("bg-[#2ED9A0]", "bg-[#FF2E7E]");
        
        // Add coloring based on 200ms threshold
        if (coreLatency < 200) {
            latencyBarFill.classList.add("bg-[#2ED9A0]"); // Mint (under limit)
        } else {
            latencyBarFill.classList.add("bg-[#FF2E7E]"); // Pink-red (over limit)
        }
    }

    if (audioBlob && recordStatus) recordStatus.textContent = "Speak Now";
}

// Fetch and display P50/P70/P100 analytics
async function updateAnalytics() {
    try {
        const res = await fetch("/api/analytics");
        if (!res.ok) throw new Error();
        
        const data = await res.json();
        
        if (data.total_runs === 0) {
            p50Val.textContent = "- ms";
            p70Val.textContent = "- ms";
            p100Val.textContent = "- ms";
            return;
        }

        p50Val.textContent = `${data.P50.rag_path.toFixed(1)} ms`;
        p70Val.textContent = `${data.P70.rag_path.toFixed(1)} ms`;
        p100Val.textContent = `${data.P100.rag_path.toFixed(1)} ms`;

        // Style Worst Case P100 dynamically
        const worstCase = data.P100.rag_path;
        p100Val.className = "text-2xl font-bold font-mono mt-1";
        if (worstCase < 200) {
            p100Val.classList.add("text-[#2ED9A0]"); // Mint (Success)
        } else if (data.P50.rag_path < 200) {
            p100Val.classList.add("text-[#FFC93C]"); // Gold (Warning)
        } else {
            p100Val.classList.add("text-[#FF2E7E]"); // Pink (Fail)
        }
    } catch (err) {
        console.error("Error fetching analytics:", err);
    }
}

// Reset analytics log trigger
if (resetBtn) {
    resetBtn.addEventListener("click", async () => {
        if (confirm("Are you sure you want to clear the latency logs?")) {
            try {
                await fetch("/api/reset_analytics", { method: "POST" });
                await updateAnalytics();
                chunksContainer.innerHTML = `<p class="text-[#9FB8AC] italic text-sm text-center py-8">No passages retrieved yet. Submit a query to see grounding details.</p>`;
                outResponse.textContent = "Pipeline responses will appear here.";
                guardrailBadges.innerHTML = "";
                outputCard.className = "lg:col-span-12 glass-card p-8 transition-all duration-300";
                
                if (translateBtn) translateBtn.classList.add("hidden");
                const voiceContainer = document.getElementById("voice-transcript-container");
                if (voiceContainer) voiceContainer.classList.add("hidden");
                originalAnswerText = "";
                translatedAnswerText = "";
                isCurrentlyTranslated = false;
                
                // Hide new sub-badges
                document.getElementById("confidence-badge").classList.add("hidden");
                document.getElementById("evidence-badge").classList.add("hidden");
                document.getElementById("evidence-container").classList.add("hidden");
                
                const timingStatusBadge = document.getElementById("timing-status-badge");
                if (timingStatusBadge) timingStatusBadge.classList.add("hidden");
                
                // Reset timing metrics text
                document.getElementById("metric-stt").textContent = "0 ms";
                document.getElementById("metric-retrieval").textContent = "0 ms";
                document.getElementById("metric-generation").textContent = "0 ms";
                document.getElementById("metric-total").textContent = "0 ms";
                
                // Reset chunk header badges
                document.getElementById("chunks-count").textContent = "0";
                document.getElementById("max-similarity-badge").classList.add("hidden");
                document.getElementById("retrieval-time-badge").classList.add("hidden");
                
                const latencyBarFill = document.getElementById("latency-bar-fill");
                if (latencyBarFill) {
                    latencyBarFill.style.width = "0%";
                    latencyBarFill.classList.remove("bg-[#2ED9A0]", "bg-[#FF2E7E]");
                }
            } catch (err) {
                console.error("Reset failed:", err);
            }
        }
    });
}

// Bind Translate Button Handler
if (translateBtn) {
    translateBtn.addEventListener("click", async () => {
        if (isCurrentlyTranslated) {
            // Restore Original Text
            outResponse.textContent = originalAnswerText;
            isCurrentlyTranslated = false;
            const isHindi = /[\u0900-\u097F]/.test(originalAnswerText);
            translateBtn.textContent = isHindi ? "Translate to English" : "Translate to Hindi";
        } else {
            if (translatedAnswerText) {
                // Show Cached Translation
                outResponse.textContent = translatedAnswerText;
                isCurrentlyTranslated = true;
                translateBtn.textContent = "Show Original";
            } else {
                const currentText = outResponse.textContent;
                const isHindi = /[\u0900-\u097F]/.test(currentText);
                const targetLang = isHindi ? "english" : "hindi";
                
                translateBtn.textContent = "Translating...";
                translateBtn.disabled = true;
                
                try {
                    const res = await fetch("/api/translate", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ text: currentText, target_lang: targetLang })
                    });
                    const data = await res.json();
                    translatedAnswerText = data.translated_text;
                    outResponse.textContent = translatedAnswerText;
                    isCurrentlyTranslated = true;
                    translateBtn.textContent = "Show Original";
                } catch (err) {
                    console.error("Translation request failed:", err);
                    translateBtn.textContent = "Error";
                    setTimeout(() => {
                        translateBtn.textContent = isHindi ? "Translate to English" : "Translate to Hindi";
                    }, 2000);
                } finally {
                    translateBtn.disabled = false;
                }
            }
        }
    });
}

// Suggested chips click binding to populate text input and trigger query
document.querySelectorAll(".query-chip").forEach(chip => {
    chip.addEventListener("click", () => {
        if (queryTextInput) {
            const text = chip.textContent.trim();
            queryTextInput.value = text;
            submitPipelineQuery(text, null);
        }
    });
});

// Setup audio and analytics on load
updateAnalytics();
