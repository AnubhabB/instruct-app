export interface Meta {
    n_tokens: number,
    n_secs: number
}

export interface QuestionAnswer {
    q: string,
    a: string,
    ts: Date,
    meta?: Meta
}

export interface Inference {
    text: string,
    meta?: Meta
}