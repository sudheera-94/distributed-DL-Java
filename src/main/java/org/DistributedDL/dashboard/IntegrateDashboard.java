package org.DistributedDL.dashboard;

import org.deeplearning4j.ui.api.UIServer;

public class IntegrateDashboard {

    public static void main(String[] args) {

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        uiServer.enableRemoteListener();

    }

}
