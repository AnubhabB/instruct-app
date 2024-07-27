<script lang="ts">
    import { invoke } from '@tauri-apps/api/tauri';
    import type { Inference, QuestionAnswer } from "$lib/types";
    import Qa from "./QA.svelte";

let qas: QuestionAnswer[] = [];
let question: string,
    asking: boolean = false,
    isrecording: boolean = false,
    recordstart: Date|null = null;

let mic: MediaStream|null = null;

const toggleRecord = async () => {
    isrecording = !isrecording;
    if(isrecording) {
        recordstart = new Date();
        mic = await navigator.mediaDevices.getUserMedia({audio: true});
    } else {
        if(mic) {
            mic.getAudioTracks().forEach(t => t.stop());
            mic = null;
        }
        if(recordstart && new Date().getTime() - recordstart.getTime() > 3000) {
            console.log("Send ask!");
        }
        recordstart = null;
    }
}

const command = async () => {
    let res: Inference = await invoke("ask", { text: qas[qas.length - 1].q });
    
    let idx = qas.length - 1;
    let qa: QuestionAnswer = qas[idx];
    qa.a = res.text;
    qa.meta = res.meta;

    qas = [...qas];

    asking = false;
}

const goAsk = async () => {
    asking = true;
    // We are just using a simple keyword to 
    qas.push({ q: question, a: "__asking__", ts: new Date() });
    question = "";

    qas = [...qas];

    // The inference generation is extremely resource intensive, giving our UI to update before the call
    setTimeout(() => {
        command()
    }, 100)
}

</script>

<div class="canvas flex flex-col">
    <div class="grid" style="grid-template-columns: 70% 30%; gap: 24px">
        <div class="input flex center relative">
            <input
                type="text"
                bind:value={question}
                on:keyup={(e) => { if(e.key == "Enter") goAsk() }}
                class="input full" placeholder="Ask your question!"
                disabled={asking}
            />
        </div>
        <div class="flex flex-row center">
            <button style="width: 96px; height: 96px; background-color: rgb(88, 117, 247); border: none; outline: none; border-radius: 50%; cursor: pointer" class="flex center justify" disabled={asking} on:click={toggleRecord}>
                {#if !isrecording}
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width=64 height=64><title>microphone</title><path fill="white" d="M12,2A3,3 0 0,1 15,5V11A3,3 0 0,1 12,14A3,3 0 0,1 9,11V5A3,3 0 0,1 12,2M19,11C19,14.53 16.39,17.44 13,17.93V21H11V17.93C7.61,17.44 5,14.53 5,11H7A5,5 0 0,0 12,16A5,5 0 0,0 17,11H19Z" /></svg>
                {:else}
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width=64 height=64><title>stop-circle-outline</title><path fill="white" d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4C16.41,4 20,7.59 20,12C20,16.41 16.41,20 12,20C7.59,20 4,16.41 4,12C4,7.59 7.59,4 12,4M9,9V15H15V9" /></svg>
                {/if}
            </button>
        </div>
    </div>
    {#each [...qas].reverse() as qa}
        <Qa qa={qa}/>
    {/each}
</div>


<style>
.canvas {
    width: 90%;
    height: 100vh;
    padding: 24px;
    max-width: 2048px;
}
</style>