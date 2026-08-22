let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordStartTime;
let timerInterval;
let chartInstance = null;
let latestAudioBlob = null; // Stably store recorded blob to prevent closure bugs

// DOM Elements
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
const outQuery = document.getElementById("out-query");
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

// Setup audio recorder
async function setupRecorder() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                audioChunks.push(event.data);
            }
        };

        mediaRecorder.onstop = async () => {
            // Securely create and store the latest audio blob
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            audioChunks = [];
            latestAudioBlob = audioBlob;
            
            // Set audio player playback source
            const audioUrl = URL.createObjectURL(latestAudioBlob);
            audioPlayback.src = audioUrl;
            audioPlayback.classList.remove("hidden");

            // Process recording and execute pipeline
            await submitPipelineQuery(null, latestAudioBlob);
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
    mediaRecorder.start();
    
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
    outQuery.textContent = text || "Audio Speech Input";
    outResponse.innerHTML = `<div class="flex items-center gap-2 text-[#FFC93C] font-semibold"><i class="fa-solid fa-spinner animate-spin"></i> Analyzing pipeline path...</div>`;
    guardrailBadges.innerHTML = "";
    
    const formData = new FormData();
    formData.append("strategy", strategySelect.value);
    formData.append("off_topic_threshold", thresholdInput.value);
    
    if (text) {
        formData.append("query_text", text);
    } else if (audioBlob) {
        formData.append("file", audioBlob, "query.wav");
    }

    try {
        const res = await fetch("/api/query", {
            method: "POST",
            body: formData
        });
        
        if (!res.ok) {
            const errDetail = await res.json();
            throw new Error(errDetail.detail || "Server error occurred");
        }
        
        const data = await res.json();
        renderPipelineResponse(data, audioBlob);
        await updateAnalytics();
    } catch (err) {
        outResponse.innerHTML = `<span class="text-[#FF2E7E] font-semibold"><i class="fa-solid fa-circle-exclamation"></i> Error: ${err.message}</span>`;
        if (audioBlob && recordStatus) recordStatus.textContent = "Query failed";
    }
}

// Render dynamic responses and badges
function renderPipelineResponse(data, audioBlob = null) {
    if (audioPlayback.classList.contains("hidden") && data.query_text) {
        outQuery.textContent = data.query_text;
    }
    
    // Status border glows
    outputCard.className = "lg:col-span-6 glass-card p-6 min-h-[220px] flex flex-col justify-between transition-all duration-300";
    if (data.status === "refused") {
        outputCard.classList.add("border-[#FF2E7E]/40", "glow-pink");
        outResponse.className = "text-[#FF2E7E] font-medium text-lg bg-black/40 p-4 rounded-xl border border-white/5 mt-2 leading-relaxed min-h-[70px]";
    } else if (data.status === "success") {
        outputCard.classList.add("border-[#2ED9A0]/40", "glow-mint");
        outResponse.className = "text-[#F3F1E7] font-medium text-lg bg-black/40 p-4 rounded-xl border border-white/5 mt-2 leading-relaxed min-h-[70px]";
    }
    outResponse.textContent = data.response_text;

    // Latency details
    const timings = data.latency_breakdown;
    const coreLatency = timings.embed_ms + timings.retrieve_ms + timings.llm_generate_ms + timings.guardrails_ms;
    outCoreLatency.textContent = `${coreLatency.toFixed(1)} ms`;
    outTotalLatency.textContent = `${timings.total_ms.toFixed(1)} ms`;

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

    // Display retrieved passages with colored strategy tags
    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
        chunksContainer.innerHTML = data.retrieved_chunks.map((chunk, idx) => {
            // Determine strategy badge style
            let strategyTag = "FIXED";
            let strategyStyle = "border-[#3B82F6] text-[#60A5FA] bg-[#3B82F6]/5";
            
            if (chunk.strategy === "sentence-aware") {
                strategyTag = "SEMANTIC";
                strategyStyle = "border-[#0D9488] text-[#2DD4BF] bg-[#0D9488]/5";
            } else if (chunk.strategy === "structure-aware") {
                strategyTag = "METADATA";
                strategyStyle = "border-[#D97706] text-[#FBBF24] bg-[#D97706]/5";
            }
            
            return `
                <div class="p-4 bg-black/40 border ${chunk.is_selected ? 'border-[#FFC93C]/50 bg-[#FFC93C]/5 shadow-[#FFC93C]/5' : 'border-white/5'} rounded-xl shadow-inner transition hover:border-white/10 duration-200">
                    <div class="flex justify-between items-center mb-2.5">
                        <div class="flex items-center gap-2">
                            <span class="px-2 py-0.5 border ${strategyStyle} rounded text-[9px] font-mono font-bold">${strategyTag}</span>
                            ${chunk.is_selected ? '<span class="px-2 py-0.5 bg-[#FFC93C]/10 text-[#FFC93C] border border-[#FFC93C]/30 rounded text-[9px] font-mono font-bold">GT SOURCE</span>' : ''}
                        </div>
                        <div class="text-right font-mono text-xs">
                            <span class="text-[#9FB8AC] mr-2">Index: ${chunk.passage_index}</span>
                            <span class="text-[#FFC93C] font-bold">Score: ${chunk.score.toFixed(3)}</span>
                        </div>
                    </div>
                    <p class="text-sm text-[#F3F1E7] leading-relaxed font-body">${chunk.text}</p>
                </div>
            `;
        }).join("");
    } else {
        chunksContainer.innerHTML = `<p class="text-[#9FB8AC] italic text-sm text-center py-8">No passages retrieved.</p>`;
    }

    // Update RAG target latency bar
    const latencyBarFill = document.getElementById("latency-bar-fill");
    if (latencyBarFill) {
        // Calculate percentage (capping at 100% for 300ms)
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

    // Refresh chart with timings
    renderLatencyChart(timings);
    if (audioBlob && recordStatus) recordStatus.textContent = "Speak Now";
}

// Draw Latency Chart matching design tokens
function renderLatencyChart(timings) {
    const ctx = document.getElementById("latencyChart").getContext("2d");
    
    const labels = ["STT", "Embed", "Search", "LLM Gen", "Guardrails"];
    const values = [
        timings.stt_ms,
        timings.embed_ms,
        timings.retrieve_ms,
        timings.llm_generate_ms,
        timings.guardrails_ms
    ];

    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: [
                    'rgba(255, 46, 126, 0.75)',  // Pink - STT
                    'rgba(255, 201, 60, 0.75)',  // Gold - Embed
                    'rgba(46, 217, 160, 0.75)',  // Mint - Search
                    'rgba(96, 165, 250, 0.75)',  // Blue - LLM Gen
                    'rgba(159, 184, 172, 0.75)'  // Muted green - Guardrails
                ],
                borderColor: [
                    '#FF2E7E', '#FFC93C', '#2ED9A0', '#60A5FA', '#9FB8AC'
                ],
                borderWidth: 1.5,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => ` ${context.parsed.x.toFixed(1)} ms`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#9FB8AC', font: { size: 9, family: 'JetBrains Mono' } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#F3F1E7', font: { weight: 'bold', size: 10 } }
                }
            }
        }
    });
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
                outQuery.textContent = "Waiting for input...";
                outResponse.textContent = "Response content will render here.";
                outCoreLatency.textContent = "- ms";
                outTotalLatency.textContent = "- ms";
                guardrailBadges.innerHTML = "";
                outputCard.className = "lg:col-span-6 glass-card p-6 min-h-[220px] flex flex-col justify-between";
                
                const latencyBarFill = document.getElementById("latency-bar-fill");
                if (latencyBarFill) {
                    latencyBarFill.style.width = "0%";
                    latencyBarFill.classList.remove("bg-[#2ED9A0]", "bg-[#FF2E7E]");
                }
                
                if (chartInstance) {
                    chartInstance.destroy();
                    chartInstance = null;
                }
            } catch (err) {
                console.error("Reset failed:", err);
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
setupRecorder();
updateAnalytics();
