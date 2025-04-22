import NextAuth from 'next-auth';
import credentialsProvider from 'next-auth/providers/credentials';
import {
  type SIWESession,
  /* verifySignature, */
  getChainIdFromMessage,
  getAddressFromMessage
} from '@reown/appkit-siwe'
import { createPublicClient, http } from 'viem'
import { users } from '../../../../server/db/schema'

import { eq } from "drizzle-orm"
import { db } from '../../../../server/db/database'

declare module 'next-auth' {
  interface Session extends SIWESession {
    address: string;
    chainId: number;
    sessionToken: string;
    userId: string;
    expires: Date;
    role: string;
  }
  
  interface User {
    role?: string;
  }
}

const nextAuthSecret = process.env.NEXTAUTH_SECRET;
if (!nextAuthSecret) {
  throw new Error('NEXTAUTH_SECRET is not set');
}

const projectId = process.env.NEXT_PUBLIC_PROJECT_ID;
if (!projectId) {
  throw new Error('NEXT_PUBLIC_PROJECT_ID is not set');
}

const providers = [
  credentialsProvider({
    name: 'Ethereum',
    credentials: {
      message: {
        label: 'Message',
        type: 'text',
        placeholder: '0x0',
      },
      signature: {
        label: 'Signature',
        type: 'text',
        placeholder: '0x0',
      },
    },
    async authorize(credentials) {
      try {
        if (!credentials?.message) {
          throw new Error('SiweMessage is undefined');
        }
        const { message, signature } = credentials;
        const address = getAddressFromMessage(message);
        const chainId = getChainIdFromMessage(message);
        console.log("authorize", { message, signature, address, chainId });

        const publicClient = createPublicClient(
          {
            transport: http(
              `https://rpc.walletconnect.org/v1/?chainId=${chainId}&projectId=${projectId}`
            )
          }
        );
        const isValid = await publicClient.verifyMessage({
          message,
          address: address as `0x${string}`,
          signature: signature as `0x${string}`
        });

        if (isValid) {
          return {
            id: `${chainId}:${address}`,
            name: chainId,
            role: 'user',
          };
        }

        return null;
      } catch (e) {
        console.error(e);
        return null;
      }
    },
  }),
];


const handler = NextAuth({
  secret: nextAuthSecret,
  providers,
  session: {
    strategy: 'jwt',
  },
  pages: {
    signIn: '/login',
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        const [, chainId, address] = user.id.split(':');
        console.log("jwt", { user, token, address, chainId });
        let userRole = 'user';
        if (address) {
          userRole = await db.select().from(users).where(eq(users.id, address)).then(rows => rows[0]?.role ?? 'user');
        }
        console.log("userRole", { userRole });

        token.sub = user.id;
        token.address = address;
        token.chainId = chainId ? parseInt(chainId, 10) : 1;
        token.role = userRole || 'user';
      }
      if (token.sub) {
        const [, chainId, address] = token.sub.split(':');
        if (chainId && address) {
          token.address = address;
          token.chainId = parseInt(chainId, 10);
          token.sessionToken = token.sub;
          token.userId = address;
        }
      }
      return token;
    },
    async signIn({ user }) {
      const [, chainId, address] = user.id.split(':');

      await db.insert(users).values({
        id: address,
        name: chainId,
        role: 'user',
      }).onConflictDoNothing({
        target: users.id,
      });
      return true;
    },

    async session({ session, token }) {
      if (token.sub) {
        const [, chainId, address] = token.sub.split(':');
        if (chainId && address) {
          session.address = address;
          session.chainId = parseInt(chainId, 10);
          session.sessionToken = token.sub;
          session.userId = address;
          session.role = typeof token.role === 'string' ? token.role : '';
        }
      }
      return session;
    },
    redirect({ url, baseUrl }) {
      if (url.startsWith("/")) return `${baseUrl}${url}`
      else if (new URL(url).origin === baseUrl) return url
      return baseUrl
    }
  },
});

export { handler as GET, handler as POST };