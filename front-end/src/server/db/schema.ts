import { pgTable, boolean, integer, timestamp, doublePrecision, index, primaryKey, text } from "drizzle-orm/pg-core"

import type { AdapterAccount } from "next-auth/adapters"

export const users = pgTable("user", {
	id: text("id")
		.primaryKey()
		.$defaultFn(() => crypto.randomUUID()),
	name: text("name"),
	email: text("email").unique(),
	emailVerified: timestamp("emailVerified", { mode: "date" }),
	image: text("image"),
	role: text("role").default("user"),
})

export const accounts = pgTable(
	"account",
	{
		userId: text("userId")
			.notNull()
			.references(() => users.id, { onDelete: "cascade" }),
		type: text("type").$type<AdapterAccount>().notNull(),
		provider: text("provider").notNull(),
		providerAccountId: text("providerAccountId").notNull(),
		refresh_token: text("refresh_token"),
		access_token: text("access_token"),
		expires_at: integer("expires_at"),
		token_type: text("token_type"),
		scope: text("scope"),
		id_token: text("id_token"),
		session_state: text("session_state"),
	},
	(account) => [
		{
			compoundKey: primaryKey({
				columns: [account.provider, account.providerAccountId],
			}),
		},
	]
)

export const sessions = pgTable("session", {
	sessionToken: text("sessionToken").primaryKey(),
	userId: text("userId")
		.notNull()
		.references(() => users.id, { onDelete: "cascade" }),
	expires: timestamp("expires", { mode: "date" }).notNull(),
})

export const verificationTokens = pgTable(
	"verificationToken",
	{
		identifier: text("identifier").notNull(),
		token: text("token").notNull(),
		expires: timestamp("expires", { mode: "date" }).notNull(),
	},
	(verificationToken) => [
		{
			compositePk: primaryKey({
				columns: [verificationToken.identifier, verificationToken.token],
			}),
		},
	]
)

export const authenticators = pgTable(
	"authenticator",
	{
		credentialID: text("credentialID").notNull().unique(),
		userId: text("userId")
			.notNull()
			.references(() => users.id, { onDelete: "cascade" }),
		providerAccountId: text("providerAccountId").notNull(),
		credentialPublicKey: text("credentialPublicKey").notNull(),
		counter: integer("counter").notNull(),
		credentialDeviceType: text("credentialDeviceType").notNull(),
		credentialBackedUp: boolean("credentialBackedUp").notNull(),
		transports: text("transports"),
	},
	(authenticator) => [
		{
			compositePK: primaryKey({
				columns: [authenticator.userId, authenticator.credentialID],
			}),
		},
	]
)

export const btc = pgTable("btc", {
	date: timestamp("date", { mode: 'string' }),
	open: doublePrecision("open"),
	high: doublePrecision("high"),
	low: doublePrecision("low"),
	close: doublePrecision("close"),
	volume: doublePrecision("volume"),
},
	(table) => {
		return {
			ixBtcDate: index("ix_btc_date").on(table.date),
		}
	});

export const modelRecommendations = pgTable("model_recommendations", {
	date: text("date"),
	model_33139: text("model_33139"),
	position: text("position"),
	side: text("side"),
	capital: text("capital"),
},
	(table) => {
		return {
			ixModelRecommendationsDate: index("ix_model_recommendations_date").on(table.date),
		}
	});

export const walletBalances = pgTable("wallet_balances", {
	height: text("height"),
	wbtc: doublePrecision("WBTC"),
	usdt: doublePrecision("USDT"),
	lgns: doublePrecision("LGNS"),
	usdPrice: doublePrecision("usdPrice"),
	blockTimestamp: timestamp("blockTimestamp", { mode: 'string' }),
	usdc: doublePrecision("USDC"),
	wmatic: doublePrecision("WMATIC"),
},
	(table) => {
		return {
			ixWalletBalancesHeight: index("ix_wallet_balances_height").on(table.height),
		}
	});

export const walletBalanceMonthly = pgTable("wallet_balance_monthly", {
	date: text("date"),
	modelo: doublePrecision("Modelo"),
	btc: doublePrecision("BTC"),
},
	(table) => {
		return {
			ixWalletBalanceMonthlyDate: index("ix_wallet_balance_monthly_date").on(table.date),
		}
	});
