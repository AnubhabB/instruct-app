<script lang="ts">
    import { invoke } from '@tauri-apps/api/tauri';
    import type { Inference, QuestionAnswer } from "$lib/types";
    import Qa from "./QA.svelte";

let qas: QuestionAnswer[] = [];
let question: string,
    asking: boolean = false;

const goAsk = async () => {
    asking = true;
    // We are just using a simple keyword to 
    qas.push({ q: question, a: "__asking__", ts: new Date() });
    question = "";

    let res: Inference = await invoke("ask", { text: qas[qas.length - 1].q });
    
    let idx = qas.length - 1;
    let qa: QuestionAnswer = qas[idx];
    qa.a = res.text;
    qa.meta = res.meta;

    qas = [...qas];

    asking = false;
}

</script>

<div class="canvas flex flex-col">
    <div class="input">
        <input
            type="text"
            bind:value={question}
            on:keyup={(e) => { if(e.key == "Enter") goAsk() }}
            class="input full" placeholder="Ask your question!"
            disabled={asking}
        />
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