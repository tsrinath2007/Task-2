let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let recordStartTime;
let timerInterval;
let chartInstance = null;

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

// Initialize listeners
thresholdInput.addEventListener("input", (e) => {
    thresholdVal.textContent = e.target.value;
});

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
            const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
            audioChunks = [];
            
            // Set audio player playback source
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioUrl;
            audioPlayback.classList.remove("hidden");

            // Process recording and execute pipeline
            await submitPipelineQuery(null, audioBlob);
        };
    } catch (err) {
        console.error("Error accessing microphone:", err);
        recordStatus.textContent = "Mic access blocked";
        recordBtn.disabled = true;
        recordBtn.classList.replace("bg-indigo-600", "bg-gray-600");
    }
}

// Start recording
function startRecording() {
    isRecording = true;
    audioChunks = [];
    mediaRecorder.start();
    
    // UI state
    recordBtn.classList.replace("bg-indigo-600", "bg-red-600");
    recordBtn.classList.add("animate-pulse");
    recordIcon.className = "fa-solid fa-square text-3xl";
    recordStatus.textContent = "Listening... Click to stop";
    recordingTime.classList.remove("hidden");
    
    // Timer
    recordStartTime = Date.now();
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordStartTime) / 1000);
        const mins = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const secs = String(elapsed % 60).padStart(2, '0');
        recordingTime.textContent = `${mins}:${secs}`;
    }, 1000);
}

// Stop recording
function stopRecording() {
    isRecording = false;
    mediaRecorder.stop();
    
    // Reset UI state
    recordBtn.classList.replace("bg-red-600", "bg-indigo-600");
    recordBtn.classList.remove("animate-pulse");
    recordIcon.className = "fa-solid fa-microphone text-4xl";
    recordStatus.textContent = "Transcribing speech...";
    recordingTime.classList.add("hidden");
    clearInterval(timerInterval);
}

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

// Submit text query
textSubmitBtn.addEventListener("click", () => {
    const text = queryTextInput.value.trim();
    if (text) {
        submitPipelineQuery(text, null);
    }
});

queryTextInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        const text = queryTextInput.value.trim();
        if (text) submitPipelineQuery(text, null);
    }
});

// API Query Trigger
async function submitPipelineQuery(text, audioBlob) {
    // Show spinner in response card
    outQuery.textContent = text || "Audio Speech Input";
    outResponse.innerHTML = `<div class="flex items-center gap-2 text-indigo-400 font-semibold"><i class="fa-solid fa-spinner animate-spin"></i> Processing pipeline...</div>`;
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
        renderPipelineResponse(data);
        await updateAnalytics();
    } catch (err) {
        outResponse.innerHTML = `<span class="text-red-500 font-semibold"><i class="fa-solid fa-circle-exclamation"></i> Error: ${err.message}</span>`;
        if (audioBlob) recordStatus.textContent = "Query failed";
    }
}

// Render dynamic responses and badges
function renderPipelineResponse(data) {
    if (audioPlayback.classList.contains("hidden") && data.query_text) {
        outQuery.textContent = data.query_text;
    }
    
    // Status color glow
    outputCard.className = "bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg min-h-[180px] flex flex-col justify-between transition-all duration-300";
    if (data.status === "refused") {
        outputCard.classList.add("glow-red", "border-red-900");
        outResponse.className = "text-red-400 font-semibold text-lg bg-gray-900/50 p-4 rounded-lg border border-red-900/30 mt-1 min-h-[60px]";
    } else if (data.status === "success") {
        outputCard.classList.add("glow-green", "border-green-900");
        outResponse.className = "text-green-300 font-semibold text-lg bg-gray-900/50 p-4 rounded-lg border border-green-900/30 mt-1 min-h-[60px]";
    }
    outResponse.textContent = data.response_text;

    // Latency details
    const timings = data.latency_breakdown;
    const coreLatency = timings.embed_ms + timings.retrieve_ms + timings.llm_generate_ms + timings.guardrails_ms;
    outCoreLatency.textContent = `${coreLatency.toFixed(1)} ms`;
    outTotalLatency.textContent = `${timings.total_ms.toFixed(1)} ms`;

    // Dynamic Guardrail Badges
    const guards = data.guardrail_results;
    let badgesHtml = "";
    
    // Safety
    if (guards.safe) {
        badgesHtml += `<span class="px-2 py-0.5 bg-green-900/30 text-green-400 border border-green-800/50 rounded-full text-xs font-semibold"><i class="fa-solid fa-shield"></i> Safe</span>`;
    } else {
        badgesHtml += `<span class="px-2 py-0.5 bg-red-900/30 text-red-400 border border-red-800/50 rounded-full text-xs font-semibold"><i class="fa-solid fa-triangle-exclamation"></i> Unsafe</span>`;
    }
    
    // Off-topic
    if (guards.off_topic) {
        badgesHtml += `<span class="px-2 py-0.5 bg-yellow-900/30 text-yellow-400 border border-yellow-800/50 rounded-full text-xs font-semibold"><i class="fa-solid fa-route"></i> Off-Topic</span>`;
    }
    
    // Groundedness
    if (data.status === "success") {
        badgesHtml += `<span class="px-2 py-0.5 bg-emerald-900/30 text-emerald-400 border border-emerald-800/50 rounded-full text-xs font-semibold"><i class="fa-solid fa-circle-check"></i> Grounded</span>`;
    } else if (guards.grounded === false && !guards.off_topic) {
        badgesHtml += `<span class="px-2 py-0.5 bg-red-900/30 text-red-400 border border-red-800/50 rounded-full text-xs font-semibold"><i class="fa-solid fa-ghost"></i> Hallucinated</span>`;
    }
    guardrailBadges.innerHTML = badgesHtml;

    // Display source passages
    if (data.retrieved_chunks && data.retrieved_chunks.length > 0) {
        chunksContainer.innerHTML = data.retrieved_chunks.map((chunk, idx) => `
            <div class="p-4 bg-gray-900/60 border ${chunk.is_selected ? 'border-indigo-500/50 bg-indigo-950/10' : 'border-gray-800'} rounded-lg">
                <div class="flex justify-between items-center mb-1 text-xs">
                    <span class="font-bold ${chunk.is_selected ? 'text-indigo-400' : 'text-gray-400'}">Passage #${idx + 1} (Idx ${chunk.passage_index})</span>
                    <div class="flex gap-2">
                        ${chunk.is_selected ? '<span class="px-1.5 py-0.2 bg-indigo-900/50 text-indigo-300 rounded font-semibold text-[10px]">Ground Truth Source</span>' : ''}
                        <span class="text-indigo-400 font-mono font-semibold">Similarity: ${chunk.score.toFixed(3)}</span>
                    </div>
                </div>
                <p class="text-sm text-gray-300 leading-relaxed">${chunk.text}</p>
            </div>
        `).join("");
    } else {
        chunksContainer.innerHTML = `<p class="text-gray-500 italic text-sm text-center py-8">No passages retrieved.</p>`;
    }

    // Refresh chart with timings
    renderLatencyChart(timings);
    if (audioBlob) recordStatus.textContent = "Done!";
}

// Draw Stacked Horizontal Chart
function renderLatencyChart(timings) {
    const ctx = document.getElementById("latencyChart").getContext("2d");
    
    // Prepare values
    const labels = ["STT", "Embed", "Retrieve", "LLM Gen", "Guardrails"];
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
                label: 'Step Latency (ms)',
                data: values,
                backgroundColor: [
                    'rgba(244, 63, 94, 0.75)',  // Rose - STT
                    'rgba(99, 102, 241, 0.75)',  // Indigo - Embed
                    'rgba(168, 85, 247, 0.75)',  // Purple - Retrieve
                    'rgba(236, 72, 153, 0.75)',  // Pink - LLM Gen
                    'rgba(16, 185, 129, 0.75)'   // Emerald - Guardrails
                ],
                borderColor: [
                    '#f43f5e', '#6366f1', '#a855f7', '#ec4899', '#10b981'
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
                    grid: { color: 'rgba(75, 85, 99, 0.2)' },
                    ticks: { color: '#9ca3af', font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e5e7eb', font: { weight: 'bold', size: 11 } }
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

        const rp = data.P50.rag_path;
        p50Val.textContent = `${data.P50.rag_path.toFixed(1)} ms`;
        p70Val.textContent = `${data.P70.rag_path.toFixed(1)} ms`;
        p100Val.textContent = `${data.P100.rag_path.toFixed(1)} ms`;

        // Check if worst-case core RAG meets latency target
        const worstCase = data.P100.rag_path;
        if (worstCase < 200) {
            p100Val.className = "font-bold text-green-400 font-mono text-sm";
        } else if (data.P50.rag_path < 200) {
            p100Val.className = "font-bold text-yellow-400 font-mono text-sm";
            p50Val.className = "font-bold text-green-400 font-mono text-sm";
        } else {
            p100Val.className = "font-bold text-red-400 font-mono text-sm";
        }
    } catch (err) {
        console.error("Error fetching analytics:", err);
    }
}

// Reset analytics log trigger
resetBtn.addEventListener("click", async () => {
    if (confirm("Are you sure you want to clear the latency logs?")) {
        try {
            await fetch("/api/reset_analytics", { method: "POST" });
            await updateAnalytics();
            chunksContainer.innerHTML = `<p class="text-gray-500 italic text-sm text-center py-8">No passages retrieved yet. Submit a query to see grounding details.</p>`;
            outQuery.textContent = "Waiting for input...";
            outResponse.textContent = "Your response will appear here.";
            outCoreLatency.textContent = "- ms";
            outTotalLatency.textContent = "- ms";
            guardrailBadges.innerHTML = "";
            outputCard.className = "bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-lg min-h-[180px] flex flex-col justify-between";
            if (chartInstance) {
                chartInstance.destroy();
                chartInstance = null;
            }
        } catch (err) {
            console.error("Reset failed:", err);
        }
    }
});

// Setup audio and analytics on load
setupRecorder();
updateAnalytics();
