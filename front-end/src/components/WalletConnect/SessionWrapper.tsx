import { useSession, SessionProvider } from "next-auth/react";
import React from "react";

function SessionWrapper({
    children,
    role,
}: {
    children: React.ReactNode;
    role: string;
}) {
    const { data: session } = useSession();
    console.log("Session data:", session);
    const isLoggedIn = !!session;
    console.log("Is logged in:", isLoggedIn);
    const isLoading = session === undefined;
    console.log("Is loading:", isLoading);
    if (isLoading) {
        return <p>Loading...</p>;
    }

    if (session) {
        if (session.role === role) {
            return children;
        } else {
            console.log("Access Denied: Invalid role");
            return <p>Access Denied: Invalid role</p>;
        }
    } else {
        return <p>Please log in to access the content.</p>;
    }
}

export default function SessionProviderWrapper({
    children,
    role,
}: {
    children: React.ReactNode;
    role: string;
}) {
    return (
        <SessionProvider>
            <SessionWrapper role={role}>{children}</SessionWrapper>
        </SessionProvider>
    );
}
