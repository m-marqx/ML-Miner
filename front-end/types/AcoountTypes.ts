interface InternalTransaction {
    transaction_hash: string;
    block_number: string;
    block_hash: string;
    type: string;
    from: string;
    to: string;
    value: string;
    gas: string;
    gas_used: string;
    input: string;
    output: string;
}

interface NftTransfer {
    token_address: string;
    token_id: string;
    from_address_entity: string | null;
    from_address_entity_logo: string | null;
    from_address: string;
    from_address_label: string | null;
    to_address_entity: string | null;
    to_address_entity_logo: string | null;
    to_address: string;
    to_address_label: string | null;
    value: string;
    amount: string;
    contract_type: string;
    block_number: string;
    block_timestamp: string;
    block_hash: string;
    transaction_hash: string;
    transaction_type: string;
    transaction_index: number;
    log_index: number;
    operator: string;
    possible_spam: string;
    verified_collection: string;
}

interface Erc20Transfer {
    token_name: string;
    token_symbol: string;
    token_logo: string | null;
    token_decimals: string;
    address: string;
    value: string;
    from_address_entity: string | null;
    from_address_entity_logo: string | null;
    from_address: string;
    from_address_label: string | null;
    to_address_entity: string | null;
    to_address_entity_logo: string | null;
    to_address: string;
    to_address_label: string | null;
    log_index: number;
    possible_spam: string | boolean;
    verified_contract: string | boolean;

    transaction_hash?: string;
    block_timestamp?: string;
    block_number?: string;
    block_hash?: string;
    transaction_index?: number;

    security_score?: number;
    direction?: string;
    value_formatted?: string;
}

interface NativeTransfer {
    from_address_entity: string | null;
    from_address_entity_logo: string | null;
    from_address: string;
    from_address_label: string | null;
    to_address_entity: string | null;
    to_address_entity_logo: string | null;
    to_address: string;
    to_address_label: string | null;
    value: string;
    value_formatted?: string;
    direction?: string;
    internal_transaction?: string;
    token_symbol?: string;
    token_logo?: string;
}

export interface TransactionHistoryItem {
    hash: string;
    nonce: string;
    transaction_index: string;
    from_address_entity: string | null;
    from_address_entity_logo: string | null;
    from_address: string;
    from_address_label: string | null;
    to_address_entity: string | null;
    to_address_entity_logo: string | null;
    to_address: string;
    to_address_label: string | null;
    value: string;
    gas: string;
    gas_price: string;
    receipt_cumulative_gas_used: string;
    receipt_gas_used: string;
    receipt_contract_address: string | null;
    receipt_status: string;
    block_timestamp: string;
    block_number: string;
    block_hash: string;

    nft_transfers: NftTransfer[];
    native_transfers: NativeTransfer[];

    receipt_root?: string;
    internal_transactions?: InternalTransaction[];
    erc20_transfer?: Erc20Transfer[];

    erc20_transfers?: Erc20Transfer[];
    transaction_fee?: string;
    method_label?: string | null;
    summary?: string;
    possible_spam?: boolean;
    category?: string;
}