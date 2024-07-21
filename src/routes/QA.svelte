<script lang="ts">
    import type { QuestionAnswer } from "$lib/types";

    export let qa: QuestionAnswer;
</script>

<div class="flex flex-col qa">
    <div class="flex flex-row caption">
        <div class="time">
            {qa.ts.toLocaleString()}
        </div>
        {#if qa.a == "__asking__"}
        <div style="color: red">Thinking ...</div>
        {/if}
        {#if qa.meta}
        <div class="flex flex-row" style="margin-left: auto; gap: 8px">
            <div>Tokens: {qa.meta.n_tokens}</div>
            <div>Time: {qa.meta.n_secs}s</div>
        </div>
        {/if}
    </div>
    <div class="question">
        {qa.q}
    </div>
    {#if qa.a != "__asking__"}
    <div class="answer">
        {qa.a}
    </div>
    {/if}
</div>

<style>
.qa {
    padding: 24px 0;
    border-bottom: 1px solid rgba(166,166,166, .75);
}

.qa .caption {
    padding: 4px 0;
}

.qa .caption * {
    font-size: 0.6rem;
}

.qa .question, .qa .answer {
    font-size: 1rem;
}

.qa .question {
    font-weight: bold;
    color: #5b8dee;
}

.qa .answer {
    padding: 8px 0;
}
</style>